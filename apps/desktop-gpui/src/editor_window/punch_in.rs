use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use cap_editor::{EditorInstance, PlaybackClockAnchor, PreviewSpeed};
use cap_project::{
    AudioTrackSegment, ProjectConfiguration, TakeVoiceSettings, TimelineConfiguration,
};
use cap_recording::{
    OggMuxer, OutputPipeline,
    feeds::microphone::{self, MicrophoneDeviceSettings, MicrophoneFeed, MicrophoneSamples},
    sources,
};
use cap_timestamp::Timestamps;
use gpui::{
    AnyElement, AppContext as _, Bounds, Context, Entity, FontWeight, Hsla, InteractiveElement,
    IntoElement, MouseButton, MouseDownEvent, ParentElement, Pixels, SharedString,
    StatefulInteractiveElement as _, Styled, Task, Window, div, prelude::FluentBuilder, px,
};
use kameo::actor::ActorRef;

use super::{EditorWindow, Selection, TrackKind, edits, markers::MARKER_SNAP_PX};
use crate::{
    editor_audio::probe_audio_duration,
    editor_side_panel::PanelTab,
    editor_sidebar::section_action,
    editor_tabs::SidebarMenu,
    editor_timeline::{
        self as timeline, PLAYHEAD_TOP_OFFSET, TIMELINE_PADDING, TRACK_GUTTER,
        clip_timeline_offsets,
    },
    feeds::Feeds,
    i18n::tr,
    recording::{RecordingOwnedFeed, setup_microphone},
    session::{Phase, RecordingSession},
    ui,
};

const PREROLL_SECS: f64 = 3.0;
const POSTROLL_SECS: f64 = 1.5;
const TAKE_EDGE_FADE_SECS: f64 = 0.015;
const START_TIMEOUT: Duration = Duration::from_secs(10);
const COUNT_IN_TICK: Duration = Duration::from_millis(50);
const MARK_HIT_PX: f32 = 4.;
const MARK_DRAG_PX: f32 = 3.;
const LATENCY_OPTIONS_MS: &[u32] = &[0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200];

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct PunchMarks {
    pub(crate) start: f64,
    pub(crate) end: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PunchMark {
    In,
    Out,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum PunchInPhase {
    #[default]
    Idle,
    Armed,
    Starting,
    Recording,
    Saving,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct PunchDisplay {
    phase: PunchInPhase,
    count: u32,
    elapsed: u32,
    left: u32,
}

#[derive(Debug, Clone, Copy)]
struct MarkDrag {
    mark: PunchMark,
    press_x: f32,
    moved: bool,
}

type ClipLayout = Vec<(u32, f64, f64, f64)>;

struct ActiveTake {
    pipeline: OutputPipeline,
    timestamps: Timestamps,
    path: String,
    _feed: Option<Arc<RecordingOwnedFeed<MicrophoneFeed>>>,
}

struct FinishedTake {
    path: String,
    duration: f64,
    timeline_start: Option<f64>,
}

struct MicInput {
    label: String,
    actor: Option<ActorRef<MicrophoneFeed>>,
    settings: Option<MicrophoneDeviceSettings>,
    meter: Option<flume::Sender<MicrophoneSamples>>,
}

#[derive(Default)]
pub(super) struct PunchInState {
    marks: PunchMarks,
    phase: PunchInPhase,
    take: Option<ActiveTake>,
    stop: Option<flume::Sender<()>>,
    drag: Option<MarkDrag>,
    clips: ClipLayout,
    generation: u64,
    lane: u32,
    lanes_before_take: Option<u32>,
    count_in_until: Option<Instant>,
    count_in_task: Option<Task<()>>,
    user_speed: Option<PreviewSpeed>,
    mic_was_released: bool,
    microphone: Option<String>,
    microphones: Vec<String>,
    meter: Option<Entity<ui::MicrophoneLevel>>,
    display: PunchDisplay,
}

impl PunchInState {
    pub(super) fn is_active(&self) -> bool {
        matches!(
            self.phase,
            PunchInPhase::Starting | PunchInPhase::Recording | PunchInPhase::Saving
        )
    }

    pub(super) fn is_armed(&self) -> bool {
        self.phase == PunchInPhase::Armed
    }

    pub(super) fn is_engaged(&self) -> bool {
        self.phase != PunchInPhase::Idle
    }

    pub(super) fn is_recording(&self) -> bool {
        self.phase == PunchInPhase::Recording
    }

    pub(super) fn is_dragging(&self) -> bool {
        self.drag.is_some()
    }

    pub(super) fn mute_recorded_mic(&self, project: &mut ProjectConfiguration, total: f64) {
        if !self.is_active() {
            return;
        }
        if let Some(timeline) = project.timeline.as_mut() {
            timeline.audio_segments.push(mic_silencer(
                self.marks.start,
                self.marks.end.unwrap_or(total),
            ));
        }
    }

    pub(super) fn finish_on_stop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.try_send(());
        }
    }
}

fn armed_marks(playhead: f64, total: f64) -> PunchMarks {
    PunchMarks {
        start: playhead.clamp(0.0, total.max(0.0)),
        end: None,
    }
}

fn with_in(marks: PunchMarks, time: f64, total: f64) -> PunchMarks {
    let start = time.clamp(0.0, total.max(0.0));
    PunchMarks {
        start,
        end: marks
            .end
            .filter(|end| end - start >= edits::MIN_AUDIO_SEGMENT_DURATION),
    }
}

fn with_out(marks: PunchMarks, time: f64, total: f64) -> Option<PunchMarks> {
    let end = time.min(total);
    (end - marks.start >= edits::MIN_AUDIO_SEGMENT_DURATION).then_some(PunchMarks {
        end: Some(end),
        ..marks
    })
}

fn dragged_marks(
    marks: PunchMarks,
    mark: PunchMark,
    time: f64,
    playhead: f64,
    snap_secs: f64,
    total: f64,
) -> PunchMarks {
    let time = if (time - playhead).abs() <= snap_secs {
        playhead
    } else {
        time
    };
    match mark {
        PunchMark::In => {
            let latest = marks
                .end
                .map_or(total, |end| end - edits::MIN_AUDIO_SEGMENT_DURATION)
                .max(0.0);
            PunchMarks {
                start: time.clamp(0.0, latest),
                ..marks
            }
        }
        PunchMark::Out => {
            let earliest = marks.start + edits::MIN_AUDIO_SEGMENT_DURATION;
            if earliest > total {
                return marks;
            }
            PunchMarks {
                end: Some(time.clamp(earliest, total)),
                ..marks
            }
        }
    }
}

fn clip_layout(timeline: Option<&TimelineConfiguration>) -> ClipLayout {
    timeline.map_or_else(Vec::new, |timeline| {
        timeline
            .segments
            .iter()
            .map(|segment| {
                (
                    segment.recording_clip,
                    segment.start,
                    segment.end,
                    segment.timescale,
                )
            })
            .collect()
    })
}

fn clamped_marks(marks: PunchMarks, total: f64) -> PunchMarks {
    let start = marks.start.clamp(0.0, total.max(0.0));
    PunchMarks {
        start,
        end: marks
            .end
            .map(|end| end.min(total))
            .filter(|end| end - start >= edits::MIN_AUDIO_SEGMENT_DURATION),
    }
}

fn marks_after_edit(
    marks: PunchMarks,
    armed_on: &ClipLayout,
    now: &ClipLayout,
    total: f64,
) -> Option<PunchMarks> {
    (armed_on == now).then(|| clamped_marks(marks, total))
}

fn preroll_start(start: f64) -> f64 {
    (start - PREROLL_SECS).max(0.0)
}

fn silent_count_in(start: f64) -> f64 {
    (PREROLL_SECS - start).max(0.0)
}

fn count_in_remaining(start: f64, playhead: f64, held: f64) -> f64 {
    (held + start - playhead).max(0.0)
}

fn live_take_span(marks: PunchMarks, playhead: f64) -> Option<(f64, f64)> {
    let end = marks.end.map_or(playhead, |end| playhead.min(end));
    (end > marks.start).then_some((marks.start, end))
}

fn take_lane(segments: &[AudioTrackSegment], marks: PunchMarks, total: f64) -> u32 {
    first_free_lane(segments, marks.start, marks.end.unwrap_or(total))
}

fn mark_time(seconds: f64) -> String {
    let centis = (seconds.max(0.0) * 100.0).round() as u64;
    let (minutes, secs, centis) = (centis / 6000, centis / 100 % 60, centis % 100);
    format!("{minutes}:{secs:02}.{centis:02}")
}

fn latency_label(ms: u32) -> String {
    if ms == 0 {
        tr::editor::window::punch_in::latency_off().to_string()
    } else {
        tr::editor::window::punch_in::latency_ms(ms).to_string()
    }
}

fn mic_silencer(start: f64, end: f64) -> AudioTrackSegment {
    AudioTrackSegment {
        volume_db: -100.0,
        replaces_mic: true,
        ..edits::default_audio_segment(start, end, 0, String::new(), String::new(), None)
    }
}

fn take_timeline_start(
    anchor: PlaybackClockAnchor,
    capture_start: Instant,
    first_sample_secs: f64,
    latency_secs: f64,
) -> Option<f64> {
    let anchor_offset = anchor
        .instant
        .checked_duration_since(capture_start)?
        .as_secs_f64();
    Some(anchor.timeline_secs + first_sample_secs - anchor_offset - latency_secs)
}

#[derive(Debug, PartialEq)]
struct TakePlacement {
    start: f64,
    end: f64,
    trim_start: f64,
}

fn place_take(
    range: PunchMarks,
    take_start: f64,
    take_duration: f64,
    total: f64,
) -> Option<TakePlacement> {
    let trim_start = (range.start - take_start).max(0.0);
    let start = take_start + trim_start;
    let recorded_end = take_start + take_duration;
    let end = range
        .end
        .unwrap_or(recorded_end)
        .min(recorded_end)
        .min(total);
    (end - start >= edits::MIN_AUDIO_SEGMENT_DURATION).then_some(TakePlacement {
        start,
        end,
        trim_start,
    })
}

fn first_free_lane(segments: &[AudioTrackSegment], start: f64, end: f64) -> u32 {
    (0..)
        .find(|lane| {
            !segments
                .iter()
                .any(|segment| segment.track == *lane && segment.start < end && segment.end > start)
        })
        .unwrap_or(0)
}

fn take_segment(take: &FinishedTake, placement: &TakePlacement, lane: u32) -> AudioTrackSegment {
    AudioTrackSegment {
        trim_start: placement.trim_start,
        fade_in: TAKE_EDGE_FADE_SECS,
        fade_out: TAKE_EDGE_FADE_SECS,
        replaces_mic: true,
        voice: Some(TakeVoiceSettings::MatchRecording),
        ..edits::default_audio_segment(
            placement.start,
            placement.end,
            lane,
            take.path.clone(),
            tr::editor::window::punch_in::take_name().to_string(),
            Some(take.duration),
        )
    }
}

fn output_time(timeline: &TimelineConfiguration, clip: u32, source: f64, near: f64) -> Option<f64> {
    let holds = timeline.hold_windows();
    timeline
        .segments
        .iter()
        .zip(clip_timeline_offsets(timeline))
        .filter(|(segment, _)| {
            segment.recording_clip == clip && (segment.start..segment.end).contains(&source)
        })
        .map(|(segment, offset)| {
            edits::effective_to_output(
                &holds,
                offset + (source - segment.start) / segment.timescale,
            )
        })
        .min_by(|left, right| (left - near).abs().total_cmp(&(right - near).abs()))
}

/// Clip trims and reorders move the recorded line without rippling the other
/// tracks, so each take is carried to where the moment under its start landed.
/// A take whose moment was cut stays put.
pub(crate) fn keep_takes_on_their_lines(
    before: &TimelineConfiguration,
    timeline: &mut TimelineConfiguration,
) {
    let moves = timeline
        .audio_segments
        .iter()
        .enumerate()
        .filter(|(_, segment)| segment.replaces_mic)
        .filter_map(|(index, segment)| {
            let (source, clip) = before.get_segment_time(segment.start)?;
            let clip = clip.recording_clip;
            let from = output_time(before, clip, source, segment.start)?;
            let to = output_time(timeline, clip, source, segment.start)?;
            Some((index, to - from))
        })
        .collect::<Vec<_>>();
    for (index, shift) in moves {
        let segment = &mut timeline.audio_segments[index];
        segment.start += shift;
        segment.end += shift;
    }
}

pub(super) fn ride_takes_with_clips(
    audio: &mut [timeline::Segment],
    takes: &[AudioTrackSegment],
    clips: &[timeline::Segment],
    shown: &[(f64, f64)],
    held_by_end: Option<usize>,
) {
    let shifts = clips
        .iter()
        .zip(shown)
        .enumerate()
        .map(|(index, (clip, (start, end)))| {
            let shift = if held_by_end == Some(index) {
                end - clip.end
            } else {
                start - clip.start
            };
            (clip.start..clip.end, shift)
        })
        .collect::<Vec<_>>();
    for (segment, take) in audio.iter_mut().zip(takes) {
        if !take.replaces_mic {
            continue;
        }
        let Some((_, shift)) = shifts
            .iter()
            .find(|(range, _)| range.contains(&segment.start))
        else {
            continue;
        };
        segment.start += shift;
        segment.end += shift;
    }
}

async fn open_mic(
    input: MicInput,
) -> Result<
    (
        Arc<microphone::MicrophoneFeedLock>,
        Option<Arc<RecordingOwnedFeed<MicrophoneFeed>>>,
    ),
    String,
> {
    if let Some(actor) = &input.actor
        && let Ok(lock) = actor.ask(microphone::Lock).await
        && lock.device_name() == input.label
    {
        return Ok((Arc::new(lock), None));
    }
    let (feed, lock, _errors) = setup_microphone(&input.label, input.settings, input.meter)
        .await
        .map_err(|error| error.to_string())?;
    Ok((lock, Some(feed)))
}

async fn start_take(project_path: PathBuf, input: MicInput) -> Result<ActiveTake, String> {
    let (lock, feed) = open_mic(input).await?;
    let audio_dir = project_path.join("assets").join("audio");
    std::fs::create_dir_all(&audio_dir).map_err(|error| error.to_string())?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let file_name = format!("take-{stamp}.ogg");
    let timestamps = Timestamps::now();
    let pipeline = OutputPipeline::builder(audio_dir.join(&file_name))
        .with_audio_source::<sources::Microphone>(lock)
        .with_timestamps(timestamps)
        .build::<OggMuxer>(Default::default())
        .await
        .map_err(|error| error.to_string())?;
    Ok(ActiveTake {
        pipeline,
        timestamps,
        path: format!("assets/audio/{file_name}"),
        _feed: feed,
    })
}

async fn finish_take(
    take: ActiveTake,
    instance: Option<Arc<EditorInstance>>,
    latency_secs: f64,
) -> Result<FinishedTake, String> {
    let finished = take
        .pipeline
        .stop()
        .await
        .map_err(|error| error.to_string())?;
    let first_sample = finished
        .first_timestamp
        .signed_duration_since_secs(take.timestamps);
    let timeline_start = instance
        .and_then(|instance| instance.last_playback_clock_anchor())
        .and_then(|anchor| {
            take_timeline_start(
                anchor,
                take.timestamps.instant(),
                first_sample,
                latency_secs,
            )
        });
    let captured = take.timestamps.instant().elapsed().as_secs_f64() - first_sample;
    let file = finished.path;
    let probed = tokio::task::spawn_blocking(move || probe_audio_duration(&file))
        .await
        .map_err(|error| error.to_string())?;
    Ok(FinishedTake {
        path: take.path,
        duration: if probed > 0.0 { probed } else { captured },
        timeline_start,
    })
}

async fn cancel_take(take: ActiveTake) {
    let path = take.pipeline.path().clone();
    let _ = take.pipeline.stop().await;
    remove_take_file(&path);
}

fn remove_take_file(path: &Path) {
    if let Err(error) = std::fs::remove_file(path) {
        tracing::warn!(%error, path = %path.display(), "removing a discarded take");
    }
}

impl EditorWindow {
    pub(crate) fn punch_in_active(&self) -> bool {
        self.punch_in.is_active()
    }

    pub(crate) fn punch_in_engaged(&self) -> bool {
        self.punch_in.is_engaged()
    }

    fn punch_in_changed(&mut self, cx: &mut Context<Self>) {
        self.timeline_view.update(cx, |_, cx| cx.notify());
        self.transport_controls.update(cx, |_, cx| cx.notify());
        self.sidebar_view.update(cx, |_, cx| cx.notify());
        cx.notify();
    }

    fn punch_in_blocked(&self) -> bool {
        !self.project_ready()
            || self.transport.is_none()
            || self.preparing_transport_active()
            || self.ordinary_handoff_frame.is_some()
            || self.crop.is_some()
            || self.busy_reason().is_some()
    }

    fn recording_session_busy(cx: &Context<Self>) -> bool {
        let session = RecordingSession::global(cx);
        let session = session.read(cx);
        session.phase != Phase::Idle || session.editor_recording_target().is_some()
    }

    pub(super) fn arm_punch_in(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.punch_in.is_engaged() || self.punch_in_blocked() {
            return;
        }
        if Self::recording_session_busy(cx) {
            self.show_notice(tr::editor::window::punch_in::recording_active(), window, cx);
            return;
        }
        if self.playing {
            self.stop_playback(cx);
        }
        self.punch_in.marks = armed_marks(self.playhead, self.total_duration());
        self.punch_in.clips = clip_layout(self.project.timeline.as_ref());
        self.punch_in.phase = PunchInPhase::Armed;
        self.sidebar.menu = None;
        self.switch_panel_tab(PanelTab::Style, window, cx);
        self.wake_punch_in_mic(window, cx);
        self.load_punch_in_microphones(cx);
        self.punch_in_changed(cx);
    }

    pub(super) fn disarm_punch_in(&mut self, cx: &mut Context<Self>) {
        if !self.punch_in.is_armed() {
            return;
        }
        self.punch_in.phase = PunchInPhase::Idle;
        self.leave_punch_in(cx);
    }

    fn leave_punch_in(&mut self, cx: &mut Context<Self>) {
        self.punch_in.drag = None;
        self.punch_in.clips.clear();
        if self.sidebar.menu.as_ref().is_some_and(|menu| {
            matches!(
                menu.kind,
                SidebarMenu::PunchInMicrophone | SidebarMenu::PunchInLatency
            )
        }) {
            self.sidebar.menu = None;
        }
        self.park_punch_in_mic(cx);
        self.punch_in_changed(cx);
    }

    pub(super) fn toggle_punch_in(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        match self.punch_in.phase {
            PunchInPhase::Idle => self.arm_punch_in(window, cx),
            PunchInPhase::Armed => self.disarm_punch_in(cx),
            PunchInPhase::Recording => self.stop_playback(cx),
            PunchInPhase::Starting | PunchInPhase::Saving => {}
        }
    }

    pub(super) fn punch_in_key(
        &mut self,
        mark: PunchMark,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match self.punch_in.phase {
            PunchInPhase::Idle => self.arm_punch_in(window, cx),
            PunchInPhase::Armed => self.set_punch_mark(mark, cx),
            _ => {}
        }
    }

    fn set_punch_mark(&mut self, mark: PunchMark, cx: &mut Context<Self>) {
        let total = self.total_duration();
        let time = self.playhead.clamp(0.0, total);
        let marks = self.punch_in.marks;
        let next = match mark {
            PunchMark::In => Some(with_in(marks, time, total)),
            PunchMark::Out => with_out(marks, time, total),
        };
        if let Some(next) = next
            && next != marks
        {
            self.punch_in.marks = next;
            self.punch_in_changed(cx);
        }
    }

    pub(super) fn sync_punch_in_to_project(&mut self, cx: &mut Context<Self>) {
        if !self.punch_in.is_armed() {
            return;
        }
        let now = clip_layout(self.project.timeline.as_ref());
        match marks_after_edit(
            self.punch_in.marks,
            &self.punch_in.clips,
            &now,
            self.total_duration(),
        ) {
            Some(marks) => {
                if marks != self.punch_in.marks {
                    self.punch_in.marks = marks;
                    self.punch_in_changed(cx);
                }
            }
            None => {
                self.disarm_punch_in(cx);
                self.post_notice(tr::editor::window::punch_in::disarmed_by_edit(), cx);
            }
        }
    }

    fn punch_mark_down(&mut self, mark: PunchMark, event: &MouseDownEvent, cx: &mut Context<Self>) {
        cx.stop_propagation();
        if !self.punch_in.is_armed() {
            return;
        }
        self.punch_in.drag = Some(MarkDrag {
            mark,
            press_x: f32::from(event.position.x),
            moved: false,
        });
    }

    pub(super) fn punch_drag_move(&mut self, x: f32, window: &mut Window, cx: &mut Context<Self>) {
        let Some(drag) = self.punch_in.drag else {
            return;
        };
        if !drag.moved && (x - drag.press_x).abs() < MARK_DRAG_PX {
            return;
        }
        let viewport_width: f32 = window.viewport_size().width.into();
        let total = self.total_duration();
        let secs_per_pixel = self
            .view
            .transform
            .secs_per_pixel(timeline::content_width(viewport_width));
        let time = timeline::time_from_x(x, viewport_width, self.view.transform, total);
        self.punch_in.marks = dragged_marks(
            self.punch_in.marks,
            drag.mark,
            time,
            self.playhead,
            MARKER_SNAP_PX * secs_per_pixel,
            total,
        );
        self.punch_in.drag = Some(MarkDrag {
            moved: true,
            ..drag
        });
        self.timeline_view.update(cx, |_, cx| cx.notify());
        self.sidebar_view.update(cx, |_, cx| cx.notify());
    }

    pub(super) fn punch_drag_up(&mut self, cx: &mut Context<Self>) {
        let Some(drag) = self.punch_in.drag.take() else {
            return;
        };
        if !drag.moved {
            let time = match drag.mark {
                PunchMark::In => Some(self.punch_in.marks.start),
                PunchMark::Out => self.punch_in.marks.end,
            };
            if let Some(time) = time {
                self.seek_to_time(time, cx);
            }
        }
        self.punch_in_changed(cx);
    }

    fn wake_punch_in_mic(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let feeds = Feeds::global(cx);
        let (released, label, level) = {
            let feeds = feeds.read(cx);
            (
                feeds.mic_input_released(),
                feeds.microphone.clone(),
                feeds.mic_level().clone(),
            )
        };
        self.punch_in.mic_was_released = released;
        self.punch_in.microphone = label.clone();
        if released && label.is_some() {
            feeds.update(cx, |feeds, cx| feeds.set_microphone(label, cx));
        }
        if self.punch_in.meter.is_none() {
            self.punch_in.meter = Some(cx.new(|cx| {
                ui::MicrophoneLevel::new(ui::MicrophoneLevelKind::DeviceRow, &level, window, cx)
            }));
        }
        self.configure_punch_in_meter(cx);
    }

    fn configure_punch_in_meter(&self, cx: &mut Context<Self>) {
        let Some(meter) = &self.punch_in.meter else {
            return;
        };
        let active = self.punch_in.is_engaged() && self.punch_in.microphone.is_some();
        let color = Hsla::from(self.theme.editor.accent);
        let track = Hsla::from(self.theme.editor.ctl_active);
        meter.update(cx, |meter, cx| meter.configure(active, color, track, cx));
    }

    fn park_punch_in_mic(&mut self, cx: &mut Context<Self>) {
        self.configure_punch_in_meter(cx);
        if !std::mem::take(&mut self.punch_in.mic_was_released)
            || Self::recording_session_busy(cx)
            || !crate::app_windows::main_window_parked(cx)
        {
            return;
        }
        Feeds::global(cx).update(cx, |feeds, cx| feeds.release_microphone(cx));
    }

    fn load_punch_in_microphones(&mut self, cx: &mut Context<Self>) {
        let listing = cx
            .background_executor()
            .spawn(async { crate::devices::list_microphones() });
        cx.spawn(async move |this, cx| {
            let microphones = listing.await;
            this.update(cx, |this, cx| {
                this.punch_in.microphones = microphones
                    .into_iter()
                    .map(|microphone| microphone.name)
                    .collect();
                this.sidebar_view.update(cx, |_, cx| cx.notify());
            })
            .ok();
        })
        .detach();
    }

    fn punch_in_microphone_options(&self) -> Vec<String> {
        let mut options = self.punch_in.microphones.clone();
        if let Some(current) = &self.punch_in.microphone
            && !options.contains(current)
        {
            options.push(current.clone());
        }
        options
    }

    pub(crate) fn punch_in_menu_items(&self, kind: SidebarMenu) -> Vec<ui::MenuItem> {
        match kind {
            SidebarMenu::PunchInLatency => {
                let current = crate::store::punch_in_latency_ms();
                LATENCY_OPTIONS_MS
                    .iter()
                    .map(|ms| ui::MenuItem::new(latency_label(*ms), *ms == current))
                    .collect()
            }
            _ => self
                .punch_in_microphone_options()
                .into_iter()
                .map(|option| {
                    let selected = self.punch_in.microphone.as_ref() == Some(&option);
                    ui::MenuItem::new(option, selected)
                })
                .collect(),
        }
    }

    pub(crate) fn choose_punch_in_menu(
        &mut self,
        kind: SidebarMenu,
        index: usize,
        cx: &mut Context<Self>,
    ) {
        if self.punch_in.is_active() {
            return;
        }
        match kind {
            SidebarMenu::PunchInLatency => {
                let Some(ms) = LATENCY_OPTIONS_MS.get(index) else {
                    return;
                };
                if !crate::store::set_store_setting(
                    crate::store::GENERAL_SETTINGS,
                    crate::store::PUNCH_IN_LATENCY_KEY,
                    serde_json::json!(ms),
                ) {
                    tracing::warn!("failed to persist the re-record latency offset");
                }
            }
            _ => {
                let Some(label) = self.punch_in_microphone_options().into_iter().nth(index) else {
                    return;
                };
                self.punch_in.microphone = Some(label.clone());
                Feeds::global(cx).update(cx, |feeds, cx| feeds.set_microphone(Some(label), cx));
                self.configure_punch_in_meter(cx);
            }
        }
        self.sidebar_view.update(cx, |_, cx| cx.notify());
        cx.notify();
    }

    pub(super) fn start_punch_in(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.punch_in.is_armed() || self.punch_in_blocked() {
            return;
        }
        if Self::recording_session_busy(cx) {
            self.show_notice(tr::editor::window::punch_in::recording_active(), window, cx);
            return;
        }
        let Some(label) = self.punch_in.microphone.clone() else {
            return;
        };
        let (actor, meter) = {
            let feeds = Feeds::global(cx);
            let feeds = feeds.read(cx);
            (feeds.mic_actor(), feeds.mic_meter())
        };
        let input = MicInput {
            settings: crate::store::RecordingDeviceSettings::for_microphone(&label),
            actor,
            meter,
            label,
        };

        if self.playing {
            self.stop_playback(cx);
        }
        let total = self.total_duration();
        self.punch_in.marks = clamped_marks(self.punch_in.marks, total);
        let marks = self.punch_in.marks;
        self.seek_to_time(preroll_start(marks.start), cx);
        self.punch_in.drag = None;
        self.punch_in.phase = PunchInPhase::Starting;
        self.punch_in.generation += 1;
        let generation = self.punch_in.generation;
        self.punch_in.lane = self.project.timeline.as_ref().map_or(0, |timeline| {
            take_lane(&timeline.audio_segments, marks, total)
        });
        if self.punch_in.lane >= self.tracks.audio {
            self.punch_in.lanes_before_take = Some(self.tracks.audio);
            self.tracks.audio = self.punch_in.lane + 1;
            self.rebuild_timeline();
        }
        self.force_take_speed();
        self.publish_project();
        self.punch_in_changed(cx);

        let project_path = self.project_path.clone();
        let task = gpui_tokio::Tokio::spawn(cx, async move {
            tokio::time::timeout(START_TIMEOUT, start_take(project_path, input))
                .await
                .unwrap_or_else(|_| Err("timed out opening the microphone".to_string()))
        });
        cx.spawn_in(window, async move |this, cx| {
            let started = task.await.unwrap_or_else(|error| Err(error.to_string()));
            this.update_in(cx, |this, window, cx| {
                this.take_started(generation, started, window, cx);
            })
            .ok();
        })
        .detach();
    }

    fn force_take_speed(&mut self) {
        if self.punch_in.user_speed.is_none() {
            self.punch_in.user_speed = Some(self.preview_speed);
        }
        self.preview_speed = PreviewSpeed::NORMAL;
        if let Some(instance) = &self.instance {
            instance.set_preview_speed(PreviewSpeed::NORMAL);
        }
    }

    fn restore_take_speed(&mut self) {
        let Some(speed) = self.punch_in.user_speed.take() else {
            return;
        };
        self.preview_speed = speed;
        if let Some(instance) = &self.instance {
            instance.set_preview_speed(speed);
        }
    }

    fn take_started(
        &mut self,
        generation: u64,
        started: Result<ActiveTake, String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if generation != self.punch_in.generation || self.punch_in.phase != PunchInPhase::Starting {
            if let Ok(take) = started {
                gpui_tokio::Tokio::spawn(cx, cancel_take(take)).detach();
            }
            return;
        }
        match started {
            Ok(take) => {
                self.punch_in.take = Some(take);
                self.punch_in.phase = PunchInPhase::Recording;
                let (stop_tx, stop_rx) = flume::bounded(1);
                self.punch_in.stop = Some(stop_tx);
                cx.spawn_in(window, async move |this, cx| {
                    if stop_rx.recv_async().await.is_ok() {
                        this.update_in(cx, |this, window, cx| this.finish_punch_in(window, cx))
                            .ok();
                    }
                })
                .detach();
                let held = silent_count_in(self.punch_in.marks.start);
                if held > 0.0 {
                    self.hold_count_in(held, generation, window, cx);
                } else {
                    self.play_take(cx);
                }
                self.punch_in.display = self.punch_in_display();
                self.punch_in_changed(cx);
            }
            Err(error) => {
                tracing::error!(%error, "starting a punch-in take");
                self.end_punch_in(cx);
                self.show_notice(
                    tr::editor::window::punch_in::start_failed(error),
                    window,
                    cx,
                );
            }
        }
    }

    fn play_take(&mut self, cx: &mut Context<Self>) {
        let marks = self.punch_in.marks;
        self.start_playback(preroll_start(marks.start), cx);
        self.play_until = marks.end.map(|end| end + POSTROLL_SECS);
    }

    fn hold_count_in(
        &mut self,
        held: f64,
        generation: u64,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.punch_in.count_in_until = Some(Instant::now() + Duration::from_secs_f64(held));
        self.punch_in.count_in_task = Some(cx.spawn_in(window, async move |this, cx| {
            loop {
                cx.background_executor().timer(COUNT_IN_TICK).await;
                let holding = this
                    .update(cx, |this, cx| {
                        if this.punch_in.generation != generation || !this.punch_in.is_recording() {
                            return false;
                        }
                        let Some(until) = this.punch_in.count_in_until else {
                            return false;
                        };
                        if Instant::now() >= until {
                            this.punch_in.count_in_until = None;
                            this.play_take(cx);
                        }
                        this.refresh_punch_in_display(cx);
                        this.punch_in.count_in_until.is_some()
                    })
                    .unwrap_or(false);
                if !holding {
                    break;
                }
            }
        }));
    }

    fn finish_punch_in(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(take) = self.punch_in.take.take() else {
            return;
        };
        self.punch_in.phase = PunchInPhase::Saving;
        self.punch_in.count_in_until = None;
        self.punch_in.count_in_task = None;
        self.punch_in_changed(cx);
        let instance = self.instance.clone();
        let latency = crate::store::punch_in_latency_secs();
        let task = gpui_tokio::Tokio::spawn(cx, finish_take(take, instance, latency));
        cx.spawn_in(window, async move |this, cx| {
            let finished = task.await.unwrap_or_else(|error| Err(error.to_string()));
            this.update_in(cx, |this, window, cx| {
                this.take_finished(finished, window, cx);
            })
            .ok();
        })
        .detach();
    }

    fn take_finished(
        &mut self,
        finished: Result<FinishedTake, String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let range = self.punch_in.marks;
        let lane = self.punch_in.lane;
        self.end_punch_in(cx);
        let take = match finished {
            Ok(take) => take,
            Err(error) => {
                tracing::error!(%error, "finishing a punch-in take");
                self.show_notice(tr::editor::window::punch_in::save_failed(error), window, cx);
                return;
            }
        };
        let placement = take
            .timeline_start
            .and_then(|start| place_take(range, start, take.duration, self.total_duration()));
        let Some(placement) = placement else {
            remove_take_file(&self.project_path.join(&take.path));
            self.show_notice(tr::editor::window::punch_in::stopped_early(), window, cx);
            return;
        };
        self.commit_take(take, placement, lane, window, cx);
    }

    pub(super) fn cancel_punch_in(&mut self, cx: &mut Context<Self>) {
        match self.punch_in.phase {
            PunchInPhase::Starting => self.end_punch_in(cx),
            PunchInPhase::Recording => {
                let take = self.punch_in.take.take();
                self.punch_in.stop = None;
                self.stop_playback(cx);
                self.end_punch_in(cx);
                if let Some(take) = take {
                    gpui_tokio::Tokio::spawn(cx, cancel_take(take)).detach();
                }
            }
            _ => {}
        }
    }

    fn end_punch_in(&mut self, cx: &mut Context<Self>) {
        let start = self.punch_in.marks.start;
        self.punch_in.phase = PunchInPhase::Idle;
        self.punch_in.stop = None;
        self.punch_in.generation += 1;
        self.punch_in.count_in_until = None;
        self.punch_in.count_in_task = None;
        self.punch_in.display = PunchDisplay::default();
        if let Some(lanes) = self.punch_in.lanes_before_take.take() {
            self.tracks.audio = lanes;
            self.rebuild_timeline();
        }
        self.restore_take_speed();
        self.publish_project();
        self.seek_to_time(start, cx);
        self.leave_punch_in(cx);
    }

    fn commit_take(
        &mut self,
        take: FinishedTake,
        placement: TakePlacement,
        lane: u32,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let segment = take_segment(&take, &placement, lane);
        let mut inserted = None;
        self.edit(
            |timeline| {
                inserted = Some(edits::insert_audio_segment(timeline, segment));
                true
            },
            window,
            cx,
        );
        let Some(index) = inserted else {
            return;
        };
        self.tracks.audio = self.tracks.audio.max(lane + 1);
        self.set_selection(Some(Selection::single(TrackKind::Audio, index)), cx);
        self.rebuild_timeline();
        self.note_edit("punch-in-take", Some(TrackKind::Audio));
        cx.notify();
        window.refresh();
    }

    fn punch_in_count_in(&self) -> f64 {
        let held = self.punch_in.count_in_until.map_or(0.0, |until| {
            until
                .saturating_duration_since(Instant::now())
                .as_secs_f64()
        });
        count_in_remaining(self.punch_in.marks.start, self.playhead, held)
    }

    fn punch_in_display(&self) -> PunchDisplay {
        let phase = self.punch_in.phase;
        if phase != PunchInPhase::Recording {
            return PunchDisplay {
                phase,
                ..PunchDisplay::default()
            };
        }
        let count = self.punch_in_count_in().ceil() as u32;
        let marks = self.punch_in.marks;
        let (elapsed, left) = if count > 0 {
            (0, 0)
        } else {
            (
                (self.playhead - marks.start).max(0.0).floor() as u32,
                marks
                    .end
                    .map_or(0, |end| (end - self.playhead).max(0.0).ceil() as u32),
            )
        };
        PunchDisplay {
            phase,
            count,
            elapsed,
            left,
        }
    }

    pub(super) fn refresh_punch_in_display(&mut self, cx: &mut Context<Self>) {
        if !self.punch_in.is_active() {
            return;
        }
        let display = self.punch_in_display();
        if display != self.punch_in.display {
            self.punch_in.display = display;
            self.sidebar_view.update(cx, |_, cx| cx.notify());
            cx.notify();
        }
    }

    pub(super) fn live_take(&self) -> Option<timeline::Segment> {
        if !self.punch_in.is_recording() || self.punch_in.count_in_until.is_some() {
            return None;
        }
        let (start, end) = live_take_span(self.punch_in.marks, self.playhead)?;
        Some(timeline::Segment {
            start,
            end,
            lane: self.punch_in.lane,
            detail: timeline::SegmentDetail::Audio {
                name: SharedString::new_static(tr::editor::window::punch_in::take_name()),
                enabled: true,
                fade_in: 0.0,
                fade_out: 0.0,
                path: SharedString::default(),
                trim_start: 0.0,
                volume_db: 0.0,
                replaces_mic: true,
                live: true,
            },
        })
    }

    pub(super) fn render_punch_marks(
        &self,
        content_width: f32,
        cx: &mut Context<Self>,
    ) -> Vec<AnyElement> {
        if !self.punch_in.is_engaged() {
            return Vec::new();
        }
        let theme = self.theme;
        let taking = self.punch_in.is_active();
        let color = if taking {
            Hsla::from(theme.red_9)
        } else {
            Hsla::from(theme.editor.accent)
        };
        let secs_per_pixel = self.view.transform.secs_per_pixel(content_width);
        let position = self.view.transform.position;
        let offset = |time: f64| ((time - position) / secs_per_pixel) as f32;
        let visible = |x: f32| (0.0..=content_width).contains(&x);
        let marks = self.punch_in.marks;
        let base = TIMELINE_PADDING + TRACK_GUTTER;

        let mut elements = Vec::new();
        let (band_end, alpha) = match marks.end {
            Some(end) => (end, if taking { 0.05 } else { 0.1 }),
            None => (self.total_duration(), 0.04),
        };
        let left = offset(marks.start).max(0.0);
        let right = offset(band_end).min(content_width);
        if right > left {
            elements.push(
                div()
                    .absolute()
                    .left(px(base + left))
                    .top(px(PLAYHEAD_TOP_OFFSET))
                    .bottom_0()
                    .w(px(right - left))
                    .bg(color.opacity(alpha))
                    .into_any_element(),
            );
        }
        let lines = [
            (
                PunchMark::In,
                Some(marks.start),
                tr::editor::window::punch_in::mark_in(),
            ),
            (
                PunchMark::Out,
                marks.end,
                tr::editor::window::punch_in::mark_out(),
            ),
        ];
        for (mark, time, label) in lines {
            let Some(x) = time.map(offset).filter(|x| visible(*x)) else {
                continue;
            };
            elements.push(self.render_punch_mark(mark, base + x, color, label, !taking, cx));
        }
        elements
    }

    fn render_punch_mark(
        &self,
        mark: PunchMark,
        x: f32,
        color: Hsla,
        label: &'static str,
        draggable: bool,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let index = mark as usize;
        let chip = div()
            .id(("punch-mark-chip", index))
            .absolute()
            .top(px(-7.))
            .map(|chip| match mark {
                PunchMark::In => chip.left(px(MARK_HIT_PX + 2.)),
                PunchMark::Out => chip.right(px(MARK_HIT_PX + 2.)),
            })
            .px(px(4.))
            .rounded(px(3.))
            .bg(color)
            .text_size(px(10.))
            .line_height(px(14.))
            .font_weight(FontWeight::MEDIUM)
            .whitespace_nowrap()
            .text_color(gpui::white())
            .child(label)
            .when(draggable, |chip| {
                chip.cursor(gpui::CursorStyle::ResizeLeftRight)
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, event: &MouseDownEvent, _window, cx| {
                            this.punch_mark_down(mark, event, cx);
                        }),
                    )
            });
        div()
            .absolute()
            .left(px(x - MARK_HIT_PX))
            .top(px(PLAYHEAD_TOP_OFFSET))
            .bottom_0()
            .w(px(MARK_HIT_PX * 2. + 1.))
            .child(
                div()
                    .absolute()
                    .left(px(MARK_HIT_PX))
                    .top_0()
                    .bottom_0()
                    .w(px(1.))
                    .bg(color),
            )
            .child(chip)
            .into_any_element()
    }

    pub(super) fn render_punch_in_overlay(&self) -> Option<AnyElement> {
        if !self.punch_in.is_recording() {
            return None;
        }
        let display = self.punch_in_display();
        if display.count > 0 {
            return Some(
                div()
                    .absolute()
                    .inset_0()
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(
                        div()
                            .size(px(120.))
                            .rounded_full()
                            .flex()
                            .items_center()
                            .justify_center()
                            .bg(gpui::hsla(0., 0., 0., 0.5))
                            .text_size(px(64.))
                            .font_weight(FontWeight::BOLD)
                            .text_color(gpui::white())
                            .child(SharedString::from(display.count.to_string())),
                    )
                    .into_any_element(),
            );
        }
        Some(
            div()
                .absolute()
                .top(px(12.))
                .right(px(12.))
                .flex()
                .flex_row()
                .items_center()
                .gap(px(6.))
                .h(px(26.))
                .px(px(10.))
                .rounded_full()
                .bg(gpui::hsla(0., 0., 0., 0.6))
                .text_size(px(12.))
                .font_weight(FontWeight::MEDIUM)
                .text_color(gpui::white())
                .child(
                    div()
                        .size(px(8.))
                        .rounded_full()
                        .bg(Hsla::from(self.theme.red_9)),
                )
                .child(tr::editor::window::punch_in::recording())
                .into_any_element(),
        )
    }

    pub(super) fn render_punch_in_button(
        &self,
        compact: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = self.theme;
        let taking = self.punch_in.is_active();
        let (tooltip, keys): (&'static str, &'static [&'static str]) = match self.punch_in.phase {
            PunchInPhase::Idle => (tr::editor::window::punch_in::tooltip_arm(), &["I", "O"]),
            PunchInPhase::Armed => (tr::editor::window::punch_in::tooltip_armed(), &["I", "O"]),
            _ => (tr::editor::window::punch_in::tooltip_stop(), &["Space"]),
        };
        let (variant, label) = match self.punch_in.phase {
            PunchInPhase::Idle => (
                ui::ButtonVariant::Gray,
                tr::editor::window::punch_in::button(),
            ),
            PunchInPhase::Armed => (
                ui::ButtonVariant::Primary,
                tr::editor::window::punch_in::button(),
            ),
            _ => (
                ui::ButtonVariant::Destructive,
                tr::editor::window::punch_in::stop(),
            ),
        };
        div()
            .id("transport-punch-in-tooltip")
            .tooltip_show_delay(ui::TOOLTIP_SHOW_DELAY)
            .tooltip(move |_, cx| {
                ui::Tooltip::new(&theme, tooltip)
                    .keys(keys.iter().copied())
                    .view(cx)
            })
            .child(
                ui::Button::plain(&theme, "transport-punch-in", variant, ui::ButtonSize::Sm)
                    .icon("icons/microphone.svg")
                    .when(!compact, |button| button.label(label))
                    .radius(px(7.))
                    .disabled(!self.project_ready() || (taking && !self.punch_in.is_recording()))
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.toggle_punch_in(window, cx);
                        window.refresh();
                    })),
            )
    }

    pub(crate) fn render_punch_in_panel(&self, cx: &mut Context<Self>) -> AnyElement {
        let theme = self.theme;
        let editor = theme.editor;
        let taking = self.punch_in.is_active();
        let hint = |text: SharedString| {
            div()
                .text_size(px(12.))
                .line_height(px(16.))
                .text_color(Hsla::from(editor.text_3))
                .child(text)
        };
        let microphone = self.punch_in.microphone.clone();
        let microphone_label = microphone
            .clone()
            .unwrap_or_else(|| tr::editor::window::punch_in::no_microphone().to_string());
        let latency = latency_label(crate::store::punch_in_latency_ms());

        let mut body = div().flex().flex_col().gap(px(14.)).child(
            ui::Field::section(&theme, tr::editor::window::punch_in::title()).child(hint(
                SharedString::new_static(tr::editor::window::punch_in::hint()),
            )),
        );
        if taking {
            body = body.child(self.render_punch_in_status(cx));
        }
        body = body
            .child(
                ui::Field::stacked(&theme, tr::editor::window::punch_in::microphone())
                    .child(
                        ui::Select::plain(&theme, "punch-in-microphone", microphone_label)
                            .stretch_label()
                            .disabled(taking)
                            .on_open(cx.listener(|this, bounds: &Bounds<Pixels>, window, cx| {
                                this.open_sidebar_menu(
                                    SidebarMenu::PunchInMicrophone,
                                    *bounds,
                                    window,
                                    cx,
                                );
                            })),
                    )
                    .children(
                        microphone
                            .and(self.punch_in.meter.clone())
                            .map(|meter| meter.into_any_element()),
                    ),
            )
            .child(self.render_punch_mark_row(PunchMark::In, taking, cx))
            .child(self.render_punch_mark_row(PunchMark::Out, taking, cx))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(4.))
                    .child(
                        ui::Subfield::plain(&theme, tr::editor::window::punch_in::latency()).child(
                            div().w(px(140.)).child(
                                ui::Select::plain(&theme, "punch-in-latency", latency)
                                    .stretch_label()
                                    .disabled(taking)
                                    .on_open(cx.listener(
                                        |this, bounds: &Bounds<Pixels>, window, cx| {
                                            this.open_sidebar_menu(
                                                SidebarMenu::PunchInLatency,
                                                *bounds,
                                                window,
                                                cx,
                                            );
                                        },
                                    )),
                            ),
                        ),
                    )
                    .child(hint(SharedString::new_static(
                        tr::editor::window::punch_in::latency_description(),
                    ))),
            );
        if !taking {
            body = body.child(
                div()
                    .flex()
                    .flex_row()
                    .gap(px(8.))
                    .child(keyed_button(
                        &theme,
                        "punch-in-record",
                        tr::editor::window::punch_in::tooltip_record(),
                        &["Shift", "Space"],
                        ui::Button::plain(
                            &theme,
                            "punch-in-record",
                            ui::ButtonVariant::Primary,
                            ui::ButtonSize::Sm,
                        )
                        .icon("icons/microphone.svg")
                        .label(tr::editor::window::punch_in::record())
                        .radius(px(7.))
                        .disabled(self.punch_in.microphone.is_none())
                        .on_click(cx.listener(|this, _, window, cx| {
                            this.start_punch_in(window, cx);
                            window.refresh();
                        })),
                    ))
                    .child(keyed_button(
                        &theme,
                        "punch-in-cancel",
                        tr::editor::window::punch_in::cancel(),
                        &["Esc"],
                        ui::Button::plain(
                            &theme,
                            "punch-in-cancel",
                            ui::ButtonVariant::Gray,
                            ui::ButtonSize::Sm,
                        )
                        .label(tr::editor::window::punch_in::cancel())
                        .radius(px(7.))
                        .on_click(cx.listener(|this, _, window, cx| {
                            this.disarm_punch_in(cx);
                            window.refresh();
                        })),
                    )),
            );
        }

        div()
            .id("punch-in-scroll")
            .block()
            .flex_1()
            .min_h_0()
            .overflow_x_hidden()
            .overflow_y_scroll()
            .text_size(px(13.))
            .child(div().pt(px(14.)).px(px(16.)).pb(px(16.)).child(body))
            .into_any_element()
    }

    fn render_punch_mark_row(
        &self,
        mark: PunchMark,
        taking: bool,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let theme = self.theme;
        let marks = self.punch_in.marks;
        let (name, value, id) = match mark {
            PunchMark::In => (
                tr::editor::window::punch_in::mark_in(),
                mark_time(marks.start),
                "punch-in-set-in",
            ),
            PunchMark::Out => (
                tr::editor::window::punch_in::mark_out(),
                marks.end.map_or_else(
                    || tr::editor::window::punch_in::until_stop().to_string(),
                    mark_time,
                ),
                "punch-in-set-out",
            ),
        };
        let mut actions = Vec::new();
        if !taking {
            actions.push(section_action(
                &theme,
                id,
                tr::editor::window::punch_in::set_to_playhead(),
                None,
                false,
                cx.listener(move |this, _, _window, cx| this.set_punch_mark(mark, cx)),
            ));
            if mark == PunchMark::Out && marks.end.is_some() {
                actions.push(section_action(
                    &theme,
                    "punch-in-clear-out",
                    tr::editor::window::punch_in::clear(),
                    None,
                    false,
                    cx.listener(|this, _, _window, cx| {
                        if this.punch_in.is_armed() {
                            this.punch_in.marks.end = None;
                            this.punch_in_changed(cx);
                        }
                    }),
                ));
            }
        }
        ui::Field::inline(&theme, name)
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .truncate()
                    .text_size(px(12.))
                    .font_features(timeline::tabular_numerals())
                    .text_color(Hsla::from(theme.editor.text_2))
                    .child(SharedString::from(value)),
            )
            .children(actions)
            .into_any_element()
    }

    fn render_punch_in_status(&self, cx: &mut Context<Self>) -> AnyElement {
        let theme = self.theme;
        let editor = theme.editor;
        let display = self.punch_in_display();
        let red = Hsla::from(theme.red_9);
        let (headline, detail): (&'static str, Option<SharedString>) = match display.phase {
            PunchInPhase::Starting => (tr::editor::window::punch_in::starting(), None),
            PunchInPhase::Saving => (tr::editor::window::punch_in::saving(), None),
            _ if display.count > 0 => (
                tr::editor::window::punch_in::count_in(),
                Some(SharedString::from(display.count.to_string())),
            ),
            _ => {
                let elapsed = tr::editor::window::punch_in::elapsed(timeline::format_time(
                    f64::from(display.elapsed),
                ));
                let detail = match self.punch_in.marks.end {
                    Some(_) => format!(
                        "{elapsed} · {}",
                        tr::editor::window::punch_in::remaining(timeline::format_time(f64::from(
                            display.left
                        )))
                    ),
                    None => elapsed.to_string(),
                };
                (
                    tr::editor::window::punch_in::recording(),
                    Some(SharedString::from(detail)),
                )
            }
        };
        let live = display.phase == PunchInPhase::Recording;
        div()
            .flex()
            .flex_col()
            .gap(px(10.))
            .p(px(12.))
            .rounded(px(10.))
            .bg(if live {
                red.opacity(0.1)
            } else {
                Hsla::from(editor.ctl)
            })
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(px(8.))
                    .child(div().size(px(8.)).flex_none().rounded_full().bg(if live {
                        red
                    } else {
                        Hsla::from(editor.text_3)
                    }))
                    .child(
                        div()
                            .text_size(px(13.))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(Hsla::from(editor.text_1))
                            .child(headline),
                    )
                    .children(detail.map(|detail| {
                        div()
                            .ml_auto()
                            .text_size(px(12.))
                            .font_features(timeline::tabular_numerals())
                            .text_color(Hsla::from(editor.text_2))
                            .child(detail)
                    })),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap(px(8.))
                    .child(keyed_button(
                        &theme,
                        "punch-in-stop",
                        tr::editor::window::punch_in::tooltip_stop(),
                        &["Space"],
                        ui::Button::plain(
                            &theme,
                            "punch-in-stop",
                            ui::ButtonVariant::Destructive,
                            ui::ButtonSize::Sm,
                        )
                        .label(tr::editor::window::punch_in::stop())
                        .radius(px(7.))
                        .disabled(!live)
                        .on_click(cx.listener(|this, _, window, cx| {
                            if this.punch_in.is_recording() {
                                this.stop_playback(cx);
                            }
                            window.refresh();
                        })),
                    ))
                    .child(keyed_button(
                        &theme,
                        "punch-in-discard",
                        tr::editor::window::punch_in::discard(),
                        &["Esc"],
                        ui::Button::plain(
                            &theme,
                            "punch-in-discard",
                            ui::ButtonVariant::Gray,
                            ui::ButtonSize::Sm,
                        )
                        .label(tr::editor::window::punch_in::discard())
                        .radius(px(7.))
                        .disabled(display.phase == PunchInPhase::Saving)
                        .on_click(cx.listener(|this, _, window, cx| {
                            this.cancel_punch_in(cx);
                            window.refresh();
                        })),
                    )),
            )
            .into_any_element()
    }
}

fn keyed_button(
    theme: &crate::theme::Theme,
    id: &'static str,
    tooltip: &'static str,
    keys: &'static [&'static str],
    button: ui::Button,
) -> impl IntoElement {
    let theme = *theme;
    div()
        .id(SharedString::from(format!("{id}-tooltip")))
        .tooltip_show_delay(ui::TOOLTIP_SHOW_DELAY)
        .tooltip(move |_, cx| {
            ui::Tooltip::new(&theme, tooltip)
                .keys(keys.iter().copied())
                .view(cx)
        })
        .child(button)
}

#[cfg(test)]
mod tests {
    use cap_project::{ClipTransition, ClipTransitionType, TimelineSegment};

    use super::*;

    const MIN: f64 = edits::MIN_AUDIO_SEGMENT_DURATION;

    fn clip(start: f64, end: f64) -> TimelineSegment {
        TimelineSegment {
            recording_clip: 0,
            start,
            end,
            timescale: 1.0,
            ..Default::default()
        }
    }

    fn timeline(
        segments: Vec<TimelineSegment>,
        takes: Vec<AudioTrackSegment>,
    ) -> TimelineConfiguration {
        TimelineConfiguration {
            segments,
            transitions: Vec::new(),
            zoom_segments: Vec::new(),
            scene_segments: Vec::new(),
            style_segments: Vec::new(),
            image_segments: Vec::new(),
            mask_segments: Vec::new(),
            text_segments: Vec::new(),
            caption_segments: Vec::new(),
            keyboard_segments: Vec::new(),
            audio_segments: takes,
            camera3d_segments: Vec::new(),
            shape_segments: Vec::new(),
        }
    }

    fn take(start: f64, end: f64) -> AudioTrackSegment {
        AudioTrackSegment {
            replaces_mic: true,
            ..edits::default_audio_segment(start, end, 0, "take.ogg".into(), "Take".into(), None)
        }
    }

    fn marks(start: f64, end: Option<f64>) -> PunchMarks {
        PunchMarks { start, end }
    }

    #[test]
    fn arming_puts_in_at_the_playhead_with_no_out() {
        assert_eq!(armed_marks(4.0, 10.0), marks(4.0, None));
        assert_eq!(armed_marks(12.0, 10.0), marks(10.0, None));
        assert_eq!(armed_marks(-1.0, 10.0), marks(0.0, None));
    }

    #[test]
    fn an_out_mark_must_land_after_in() {
        let armed = marks(4.0, None);
        assert_eq!(with_out(armed, 2.0, 10.0), None);
        assert_eq!(with_out(armed, 4.0 + MIN / 2.0, 10.0), None);
        assert_eq!(with_out(armed, 6.0, 10.0), Some(marks(4.0, Some(6.0))));
    }

    #[test]
    fn moving_in_past_out_clears_out() {
        let set = marks(4.0, Some(6.0));
        assert_eq!(with_in(set, 5.0, 10.0), marks(5.0, Some(6.0)));
        assert_eq!(with_in(set, 6.0, 10.0), marks(6.0, None));
        assert_eq!(with_in(set, 8.0, 10.0), marks(8.0, None));
    }

    #[test]
    fn dragged_marks_keep_their_order_and_stay_on_the_timeline() {
        let set = marks(4.0, Some(6.0));
        assert_eq!(
            dragged_marks(set, PunchMark::In, 7.0, 0.0, 0.0, 10.0),
            marks(6.0 - MIN, Some(6.0))
        );
        assert_eq!(
            dragged_marks(set, PunchMark::In, -3.0, 0.0, 0.0, 10.0),
            marks(0.0, Some(6.0))
        );
        assert_eq!(
            dragged_marks(set, PunchMark::Out, 1.0, 0.0, 0.0, 10.0),
            marks(4.0, Some(4.0 + MIN))
        );
        assert_eq!(
            dragged_marks(set, PunchMark::Out, 15.0, 0.0, 0.0, 10.0),
            marks(4.0, Some(10.0))
        );
        assert_eq!(
            dragged_marks(marks(4.0, None), PunchMark::In, 12.0, 0.0, 0.0, 10.0),
            marks(10.0, None)
        );
    }

    #[test]
    fn a_dragged_mark_snaps_to_a_nearby_playhead() {
        let set = marks(4.0, Some(6.0));
        assert_eq!(
            dragged_marks(set, PunchMark::In, 3.05, 3.0, 0.1, 10.0),
            marks(3.0, Some(6.0))
        );
        assert_eq!(
            dragged_marks(set, PunchMark::In, 3.5, 3.0, 0.1, 10.0),
            marks(3.5, Some(6.0))
        );
        assert_eq!(
            dragged_marks(set, PunchMark::Out, 7.95, 8.0, 0.1, 10.0),
            marks(4.0, Some(8.0))
        );
    }

    #[test]
    fn out_cannot_be_dragged_when_in_sits_at_the_end() {
        let set = marks(10.0, None);
        assert_eq!(dragged_marks(set, PunchMark::Out, 9.0, 0.0, 0.0, 10.0), set);
    }

    #[test]
    fn the_count_in_is_always_three_seconds() {
        for start in [0.0, 0.5, 2.0, 3.0, 8.0] {
            let held = silent_count_in(start);
            let from = preroll_start(start);
            assert!((count_in_remaining(start, from, held) - PREROLL_SECS).abs() < 1e-9);
        }
        assert_eq!(silent_count_in(0.5), 2.5);
        assert_eq!(preroll_start(0.5), 0.0);
        assert_eq!(silent_count_in(5.0), 0.0);
        assert_eq!(preroll_start(5.0), 2.0);
        assert_eq!(count_in_remaining(5.0, 6.0, 0.0), 0.0);
    }

    #[test]
    fn a_clip_layout_change_disarms_and_a_shorter_timeline_clamps() {
        let armed_on = clip_layout(Some(&timeline(vec![clip(0.0, 10.0)], Vec::new())));
        let trimmed = clip_layout(Some(&timeline(vec![clip(0.0, 8.0)], Vec::new())));
        assert_eq!(
            marks_after_edit(marks(4.0, Some(6.0)), &armed_on, &trimmed, 8.0),
            None
        );
        assert_eq!(
            marks_after_edit(marks(4.0, Some(9.0)), &armed_on, &armed_on, 8.0),
            Some(marks(4.0, Some(8.0)))
        );
        assert_eq!(
            marks_after_edit(marks(9.0, Some(9.5)), &armed_on, &armed_on, 8.0),
            Some(marks(8.0, None))
        );
    }

    #[test]
    fn the_live_take_grows_from_in_and_stops_at_out() {
        let set = marks(4.0, Some(6.0));
        assert_eq!(live_take_span(set, 3.0), None);
        assert_eq!(live_take_span(set, 4.0), None);
        assert_eq!(live_take_span(set, 5.0), Some((4.0, 5.0)));
        assert_eq!(live_take_span(set, 7.0), Some((4.0, 6.0)));
        assert_eq!(live_take_span(marks(4.0, None), 9.0), Some((4.0, 9.0)));
    }

    #[test]
    fn the_live_take_lane_is_the_lane_the_take_commits_to() {
        let mut upper = take(4.0, 6.0);
        upper.track = 1;
        let segments = vec![take(0.0, 5.0), upper];
        assert_eq!(take_lane(&segments, marks(4.5, Some(5.5)), 10.0), 2);
        assert_eq!(take_lane(&segments, marks(5.5, Some(7.0)), 10.0), 0);
        assert_eq!(take_lane(&segments, marks(3.0, None), 10.0), 2);
        let placement = place_take(marks(4.5, Some(5.5)), 2.0, 5.0, 10.0).unwrap();
        assert_eq!(
            first_free_lane(&segments, placement.start, placement.end),
            take_lane(&segments, marks(4.5, Some(5.5)), 10.0)
        );
    }

    #[test]
    fn mark_times_read_to_the_hundredth() {
        assert_eq!(mark_time(4.256), "0:04.26");
        assert_eq!(mark_time(65.0), "1:05.00");
    }

    #[test]
    fn the_take_covers_the_punch_range_and_keeps_the_preroll_as_a_handle() {
        assert_eq!(
            place_take(marks(5.0, Some(7.0)), 2.0, 7.0, 20.0),
            Some(TakePlacement {
                start: 5.0,
                end: 7.0,
                trim_start: 3.0
            })
        );
    }

    #[test]
    fn a_take_that_began_during_a_silent_count_in_is_trimmed_to_in() {
        assert_eq!(
            place_take(marks(1.0, None), -2.0, 6.0, 20.0),
            Some(TakePlacement {
                start: 1.0,
                end: 4.0,
                trim_start: 3.0
            })
        );
    }

    #[test]
    fn an_open_ended_take_runs_to_where_recording_stopped() {
        let placement = place_take(marks(5.0, None), 2.0, 7.0, 20.0).unwrap();
        assert_eq!(placement.end, 9.0);
    }

    #[test]
    fn a_take_stopped_before_the_punch_in_point_is_not_placed() {
        assert_eq!(place_take(marks(5.0, None), 2.0, 2.0, 20.0), None);
    }

    #[test]
    fn the_playback_clock_maps_the_first_sample_onto_the_timeline() {
        let capture_start = Instant::now();
        let anchor = PlaybackClockAnchor {
            instant: capture_start + Duration::from_millis(500),
            timeline_secs: 2.0,
        };
        let start = take_timeline_start(anchor, capture_start, 0.1, 0.0).unwrap();
        assert!((start - 1.6).abs() < 1e-9);
    }

    #[test]
    fn the_latency_setting_moves_the_take_earlier() {
        let capture_start = Instant::now();
        let anchor = PlaybackClockAnchor {
            instant: capture_start + Duration::from_millis(500),
            timeline_secs: 2.0,
        };
        let start = take_timeline_start(anchor, capture_start, 0.1, 0.04).unwrap();
        assert!((start - 1.56).abs() < 1e-9);
    }

    #[test]
    fn a_clock_from_before_the_take_cannot_place_it() {
        let capture_start = Instant::now() + Duration::from_secs(1);
        let anchor = PlaybackClockAnchor {
            instant: Instant::now(),
            timeline_secs: 2.0,
        };
        assert_eq!(take_timeline_start(anchor, capture_start, 0.1, 0.0), None);
    }

    #[test]
    fn a_take_lands_on_the_lowest_lane_free_across_its_range() {
        let mut upper = take(4.0, 6.0);
        upper.track = 1;
        let segments = vec![take(0.0, 5.0), upper];
        assert_eq!(first_free_lane(&segments, 5.5, 7.0), 0);
        assert_eq!(first_free_lane(&segments, 4.5, 5.5), 2);
    }

    #[test]
    fn the_silencer_replaces_the_mic_below_the_decode_cutoff() {
        let silencer = mic_silencer(1.0, 2.0);
        assert!(silencer.replaces_mic);
        assert!(silencer.volume_db <= -60.0);
    }

    #[test]
    fn a_take_follows_its_line_when_a_clip_before_it_is_trimmed() {
        let before = timeline(vec![clip(0.0, 10.0)], vec![take(6.0, 7.0)]);
        let mut after = before.clone();
        after.segments[0].start = 2.0;
        keep_takes_on_their_lines(&before, &mut after);
        assert!((after.audio_segments[0].start - 4.0).abs() < 1e-9);
        assert!((after.audio_segments[0].end - 5.0).abs() < 1e-9);
    }

    #[test]
    fn a_take_follows_its_clip_when_clips_are_reordered() {
        let before = timeline(vec![clip(0.0, 4.0), clip(10.0, 12.0)], vec![take(1.0, 2.0)]);
        let mut after = before.clone();
        after.segments.reverse();
        keep_takes_on_their_lines(&before, &mut after);
        assert!((after.audio_segments[0].start - 3.0).abs() < 1e-9);
    }

    #[test]
    fn a_take_whose_moment_was_cut_stays_put() {
        let before = timeline(vec![clip(0.0, 10.0)], vec![take(1.0, 2.0)]);
        let mut after = before.clone();
        after.segments[0].start = 3.0;
        keep_takes_on_their_lines(&before, &mut after);
        assert_eq!(after.audio_segments[0].start, 1.0);
    }

    fn crossfaded_timeline() -> TimelineConfiguration {
        let mut second = clip(0.0, 10.0);
        second.recording_clip = 1;
        let mut timeline = timeline(vec![clip(0.0, 10.0), second], vec![take(14.0, 15.0)]);
        timeline.transitions.push(ClipTransition {
            segment_index: 1,
            kind: ClipTransitionType::CrossFade,
            duration: 1.0,
        });
        timeline
    }

    #[test]
    fn an_unchanged_timeline_never_moves_a_take_past_a_transition() {
        let before = crossfaded_timeline();
        let mut after = before.clone();
        keep_takes_on_their_lines(&before, &mut after);
        assert_eq!(after.audio_segments[0].start, 14.0);
    }

    #[test]
    fn trimming_after_a_take_past_a_transition_leaves_it_alone() {
        let before = crossfaded_timeline();
        let mut after = before.clone();
        after.segments[1].end = 9.0;
        keep_takes_on_their_lines(&before, &mut after);
        assert_eq!(after.audio_segments[0].start, 14.0);
    }

    #[test]
    fn a_take_past_a_transition_follows_a_trim_before_it() {
        let before = crossfaded_timeline();
        let mut after = before.clone();
        after.segments[0].start = 2.0;
        keep_takes_on_their_lines(&before, &mut after);
        assert!((after.audio_segments[0].start - 12.0).abs() < 1e-9);
    }

    fn held_timeline() -> TimelineConfiguration {
        let mut timeline: TimelineConfiguration = serde_json::from_value(serde_json::json!({
            "segments": [
                { "recordingSegment": 0, "timescale": 1.0, "start": 0.0, "end": 10.0 }
            ],
            "zoomSegments": [],
            "textSegments": [
                { "start": 2.0, "end": 4.0, "track": 0, "content": "x", "layout": "fullscreen" }
            ]
        }))
        .unwrap();
        timeline.audio_segments = vec![take(3.0, 3.5), take(6.0, 7.0)];
        timeline
    }

    #[test]
    fn takes_around_a_text_hold_stay_put_when_nothing_moved() {
        let before = held_timeline();
        let mut after = before.clone();
        keep_takes_on_their_lines(&before, &mut after);
        assert_eq!(after.audio_segments[0].start, 3.0);
        assert_eq!(after.audio_segments[1].start, 6.0);
    }

    #[test]
    fn a_take_after_a_text_hold_follows_a_trim_before_it() {
        let before = held_timeline();
        let mut after = before.clone();
        after.segments[0].end = 9.0;
        keep_takes_on_their_lines(&before, &mut after);
        assert_eq!(after.audio_segments[1].start, 6.0);

        after = before.clone();
        after.segments[0].start = 1.0;
        keep_takes_on_their_lines(&before, &mut after);
        assert!((after.audio_segments[1].start - 5.0).abs() < 1e-9);
    }

    fn drawn(start: f64, end: f64) -> timeline::Segment {
        timeline::Segment {
            start,
            end,
            lane: 0,
            detail: timeline::SegmentDetail::Caption { text: "".into() },
        }
    }

    #[test]
    fn takes_are_drawn_with_the_clip_box_they_start_in() {
        let clips = [drawn(0.0, 4.0), drawn(4.0, 10.0)];
        let mut music = take(5.0, 6.0);
        music.replaces_mic = false;
        let takes = [take(1.0, 2.0), take(5.0, 6.0), music];
        let mut audio = takes
            .iter()
            .map(|take| drawn(take.start, take.end))
            .collect::<Vec<_>>();
        ride_takes_with_clips(&mut audio, &takes, &clips, &[(0.0, 4.0), (5.5, 11.5)], None);
        assert_eq!((audio[0].start, audio[0].end), (1.0, 2.0));
        assert_eq!((audio[1].start, audio[1].end), (6.5, 7.5));
        assert_eq!(audio[2].start, 5.0);
    }

    #[test]
    fn a_start_edge_drag_moves_takes_with_the_clip_end() {
        let clips = [drawn(0.0, 4.0)];
        let takes = [take(1.0, 2.0)];
        let mut audio = vec![drawn(1.0, 2.0)];
        ride_takes_with_clips(&mut audio, &takes, &clips, &[(0.0, 5.0)], Some(0));
        assert_eq!(audio[0].start, 2.0);
        ride_takes_with_clips(&mut audio, &takes, &clips, &[(1.0, 4.0)], Some(0));
        assert_eq!(audio[0].start, 2.0);
    }

    #[test]
    fn music_does_not_follow_clip_edits() {
        let mut music = take(6.0, 7.0);
        music.replaces_mic = false;
        let before = timeline(vec![clip(0.0, 10.0)], vec![music]);
        let mut after = before.clone();
        after.segments[0].start = 2.0;
        keep_takes_on_their_lines(&before, &mut after);
        assert_eq!(after.audio_segments[0].start, 6.0);
    }
}
