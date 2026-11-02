use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use cap_editor::{EditorInstance, PlaybackClockAnchor};
use cap_project::{
    AudioTrackSegment, ProjectConfiguration, TakeVoiceSettings, TimelineConfiguration,
};
use cap_recording::{
    OggMuxer, OutputPipeline,
    feeds::microphone::{self, MicrophoneDeviceSettings, MicrophoneFeed, MicrophoneSamples},
    sources,
};
use cap_timestamp::Timestamps;
use gpui::{Context, Window};
use kameo::actor::ActorRef;

use super::{EditorWindow, Selection, TrackKind, edits};
use crate::{
    editor_audio::probe_audio_duration,
    editor_timeline::{self as timeline, clip_timeline_offsets},
    feeds::Feeds,
    i18n::tr,
    recording::{RecordingOwnedFeed, setup_microphone},
    session::{Phase, RecordingSession},
};

const PREROLL_SECS: f64 = 3.0;
const POSTROLL_SECS: f64 = 1.5;
const TAKE_EDGE_FADE_SECS: f64 = 0.015;
const START_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct PunchMarks {
    pub(crate) start: Option<f64>,
    pub(crate) end: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum PunchMark {
    In,
    Out,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct PunchRange {
    start: f64,
    end: Option<f64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
enum PunchInPhase {
    #[default]
    Idle,
    Starting,
    Recording,
    Saving,
}

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
    pub(super) marks: PunchMarks,
    phase: PunchInPhase,
    range: PunchRange,
    take: Option<ActiveTake>,
    stop: Option<flume::Sender<()>>,
}

impl PunchInState {
    pub(super) fn is_active(&self) -> bool {
        self.phase != PunchInPhase::Idle
    }

    pub(super) fn is_recording(&self) -> bool {
        self.phase == PunchInPhase::Recording
    }

    pub(super) fn is_transitioning(&self) -> bool {
        matches!(self.phase, PunchInPhase::Starting | PunchInPhase::Saving)
    }

    pub(super) fn mute_recorded_mic(&self, project: &mut ProjectConfiguration, total: f64) {
        if !self.is_active() {
            return;
        }
        if let Some(timeline) = project.timeline.as_mut() {
            timeline.audio_segments.push(mic_silencer(
                self.range.start,
                self.range.end.unwrap_or(total),
            ));
        }
    }

    pub(super) fn finish_on_stop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.try_send(());
        }
    }

    fn set_mark(&mut self, mark: PunchMark, time: f64) {
        let slot = match mark {
            PunchMark::In => &mut self.marks.start,
            PunchMark::Out => &mut self.marks.end,
        };
        *slot = (!slot.is_some_and(|current| (current - time).abs() < 0.01)).then_some(time);
    }
}

fn punch_range(marks: PunchMarks, playhead: f64, total: f64) -> PunchRange {
    let start = marks.start.unwrap_or(playhead).clamp(0.0, total);
    let end = marks
        .end
        .filter(|end| *end > start)
        .map(|end| end.min(total));
    PunchRange { start, end }
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
) -> Option<f64> {
    let anchor_offset = anchor
        .instant
        .checked_duration_since(capture_start)?
        .as_secs_f64();
    Some(anchor.timeline_secs + first_sample_secs - anchor_offset)
}

#[derive(Debug, PartialEq)]
struct TakePlacement {
    start: f64,
    end: f64,
    trim_start: f64,
}

fn place_take(
    range: PunchRange,
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
        .and_then(|anchor| take_timeline_start(anchor, take.timestamps.instant(), first_sample));
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

    fn punch_in_changed(&mut self, cx: &mut Context<Self>) {
        self.timeline_view.update(cx, |_, cx| cx.notify());
        self.transport_controls.update(cx, |_, cx| cx.notify());
        cx.notify();
    }

    pub(super) fn set_punch_mark(&mut self, mark: PunchMark, cx: &mut Context<Self>) {
        if self.punch_in.is_active() || !self.project_ready() {
            return;
        }
        let time = self.playhead.clamp(0.0, self.total_duration());
        self.punch_in.set_mark(mark, time);
        self.punch_in_changed(cx);
    }

    pub(super) fn toggle_punch_in(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.punch_in.is_recording() {
            self.stop_playback(cx);
        } else if !self.punch_in.is_active() {
            self.start_punch_in(window, cx);
        }
    }

    fn start_punch_in(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.project_ready()
            || self.transport.is_none()
            || self.preparing_transport_active()
            || self.ordinary_handoff_frame.is_some()
            || self.crop.is_some()
            || self.busy_reason().is_some()
        {
            return;
        }
        let session = RecordingSession::global(cx);
        if session.read(cx).phase != Phase::Idle
            || session.read(cx).editor_recording_target().is_some()
        {
            self.show_notice(tr::editor::window::punch_in::recording_active(), window, cx);
            return;
        }
        let feeds = Feeds::global(cx);
        let (label, actor, meter) = {
            let feeds = feeds.read(cx);
            (
                feeds.microphone.clone(),
                feeds.mic_actor(),
                feeds.mic_meter(),
            )
        };
        let Some(label) = label else {
            self.show_notice(tr::editor::window::punch_in::no_microphone(), window, cx);
            return;
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
        self.punch_in.range =
            punch_range(self.punch_in.marks, self.playhead, self.total_duration());
        self.punch_in.phase = PunchInPhase::Starting;
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
                this.take_started(started, window, cx);
            })
            .ok();
        })
        .detach();
    }

    fn take_started(
        &mut self,
        started: Result<ActiveTake, String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match started {
            Ok(take) => {
                self.punch_in.take = Some(take);
                self.punch_in.phase = PunchInPhase::Recording;
                let (stop_tx, stop_rx) = flume::bounded(1);
                self.punch_in.stop = Some(stop_tx);
                let range = self.punch_in.range;
                self.start_playback((range.start - PREROLL_SECS).max(0.0), cx);
                self.play_until = range.end.map(|end| end + POSTROLL_SECS);
                cx.spawn_in(window, async move |this, cx| {
                    if stop_rx.recv_async().await.is_ok() {
                        this.update_in(cx, |this, window, cx| this.finish_punch_in(window, cx))
                            .ok();
                    }
                })
                .detach();
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

    fn finish_punch_in(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(take) = self.punch_in.take.take() else {
            return;
        };
        self.punch_in.phase = PunchInPhase::Saving;
        self.punch_in_changed(cx);
        let instance = self.instance.clone();
        let task = gpui_tokio::Tokio::spawn(cx, finish_take(take, instance));
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
        let range = self.punch_in.range;
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
        self.commit_take(take, placement, window, cx);
    }

    pub(super) fn cancel_punch_in(&mut self, cx: &mut Context<Self>) {
        let Some(take) = self.punch_in.take.take() else {
            return;
        };
        self.punch_in.stop = None;
        self.stop_playback(cx);
        self.end_punch_in(cx);
        gpui_tokio::Tokio::spawn(cx, cancel_take(take)).detach();
    }

    fn end_punch_in(&mut self, cx: &mut Context<Self>) {
        self.punch_in.phase = PunchInPhase::Idle;
        self.punch_in.stop = None;
        self.publish_project();
        self.punch_in_changed(cx);
    }

    fn commit_take(
        &mut self,
        take: FinishedTake,
        placement: TakePlacement,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let lane = self.project.timeline.as_ref().map_or(0, |timeline| {
            first_free_lane(&timeline.audio_segments, placement.start, placement.end)
        });
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

    pub(super) fn punch_in_label(&self) -> String {
        if self.punch_in.is_recording() {
            let remaining = self.punch_in.range.start - self.playhead;
            if remaining > 0.0 {
                return format!("{remaining:.1}");
            }
            return tr::editor::window::punch_in::stop().to_string();
        }
        match self.punch_in.phase {
            PunchInPhase::Starting => tr::editor::window::punch_in::starting().to_string(),
            PunchInPhase::Saving => tr::editor::window::punch_in::saving().to_string(),
            _ => tr::editor::window::punch_in::button().to_string(),
        }
    }

    pub(super) fn punch_in_tooltip(&self) -> String {
        if self.punch_in.is_recording() {
            return tr::editor::window::punch_in::tooltip_stop().to_string();
        }
        let range = punch_range(self.punch_in.marks, self.playhead, self.total_duration());
        let start = timeline::format_time(range.start);
        match range.end {
            Some(end) => {
                tr::editor::window::punch_in::tooltip_range(start, timeline::format_time(end))
            }
            None => tr::editor::window::punch_in::tooltip_from(start),
        }
    }
}

#[cfg(test)]
mod tests {
    use cap_project::{ClipTransition, ClipTransitionType, TimelineSegment};

    use super::*;

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

    #[test]
    fn the_range_runs_from_the_marks_or_the_playhead() {
        let marks = PunchMarks {
            start: Some(4.0),
            end: Some(6.0),
        };
        assert_eq!(
            punch_range(marks, 1.0, 10.0),
            PunchRange {
                start: 4.0,
                end: Some(6.0)
            }
        );
        assert_eq!(
            punch_range(PunchMarks::default(), 1.0, 10.0),
            PunchRange {
                start: 1.0,
                end: None
            }
        );
    }

    #[test]
    fn an_out_mark_before_the_in_point_leaves_the_take_open_ended() {
        let marks = PunchMarks {
            start: Some(4.0),
            end: Some(2.0),
        };
        assert_eq!(punch_range(marks, 0.0, 10.0).end, None);
    }

    #[test]
    fn setting_a_mark_twice_at_the_same_spot_clears_it() {
        let mut state = PunchInState::default();
        state.set_mark(PunchMark::In, 2.0);
        assert_eq!(state.marks.start, Some(2.0));
        state.set_mark(PunchMark::In, 3.0);
        assert_eq!(state.marks.start, Some(3.0));
        state.set_mark(PunchMark::In, 3.0);
        assert_eq!(state.marks.start, None);
    }

    #[test]
    fn the_take_covers_the_punch_range_and_keeps_the_preroll_as_a_handle() {
        let range = PunchRange {
            start: 5.0,
            end: Some(7.0),
        };
        assert_eq!(
            place_take(range, 2.0, 7.0, 20.0),
            Some(TakePlacement {
                start: 5.0,
                end: 7.0,
                trim_start: 3.0
            })
        );
    }

    #[test]
    fn an_open_ended_take_runs_to_where_recording_stopped() {
        let range = PunchRange {
            start: 5.0,
            end: None,
        };
        let placement = place_take(range, 2.0, 7.0, 20.0).unwrap();
        assert_eq!(placement.end, 9.0);
    }

    #[test]
    fn a_take_stopped_before_the_punch_in_point_is_not_placed() {
        let range = PunchRange {
            start: 5.0,
            end: None,
        };
        assert_eq!(place_take(range, 2.0, 2.0, 20.0), None);
    }

    #[test]
    fn the_playback_clock_maps_the_first_sample_onto_the_timeline() {
        let capture_start = Instant::now();
        let anchor = PlaybackClockAnchor {
            instant: capture_start + Duration::from_millis(500),
            timeline_secs: 2.0,
        };
        let start = take_timeline_start(anchor, capture_start, 0.1).unwrap();
        assert!((start - 1.6).abs() < 1e-9);
    }

    #[test]
    fn a_clock_from_before_the_take_cannot_place_it() {
        let capture_start = Instant::now() + Duration::from_secs(1);
        let anchor = PlaybackClockAnchor {
            instant: Instant::now(),
            timeline_secs: 2.0,
        };
        assert_eq!(take_timeline_start(anchor, capture_start, 0.1), None);
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
