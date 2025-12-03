import os
import sys
import pytest

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.analyzer import analyze_video


def test_analyze_video_creates_output(tmp_path):
    input_video = "sample_dance.mp4"
    output_video_path = tmp_path / "test_output.mp4"

    if not os.path.exists(input_video):
        pytest.skip("sample_dance.mp4 not found, skipping test")

    result_path = analyze_video(input_video, str(output_video_path))

    assert os.path.exists(result_path)
    assert os.path.getsize(result_path) > 0


def test_analyze_video_raises_for_missing_input():
    fake_input = "this_file_does_not_exist.mp4"
    output_path = "should_not_be_created.mp4"

    with pytest.raises(FileNotFoundError):
        analyze_video(fake_input, output_path)
