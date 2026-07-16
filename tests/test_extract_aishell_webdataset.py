import io
from pathlib import Path
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extract_aishell_webdataset import extract_wavs, read_speaker_info


def _member(archive, name, content):
  info = tarfile.TarInfo(name)
  info.size = len(content)
  archive.addfile(info, io.BytesIO(content))


def test_extracts_only_requested_gender_and_is_resumable():
  with tempfile.TemporaryDirectory() as directory:
    base = Path(directory)
    speaker_info = base / 'speaker.info'
    speaker_info.write_text('0002 M\n0124 F\n', encoding='utf-8')
    archive_path = base / 'train_000000.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
      _member(archive, 'BAC009S0002W0001.wav', b'male')
      _member(archive, 'BAC009S0124W0001.wav', b'female')
      _member(archive, 'BAC009S0124W0001.json', b'{"text":"unused"}')

    output = base / 'wav'
    first = extract_wavs([archive_path], speaker_info, output)
    assert first['totals'] == {
        'gender_excluded': 1, 'json_ignored': 1, 'wav_extracted': 1}
    assert first['speakers'] == {'S0124': 1}
    assert (output / 'S0124' / 'BAC009S0124W0001.wav').read_bytes() == b'female'
    assert not (output / 'S0002').exists()
    assert not list(output.rglob('*.json'))

    second = extract_wavs([archive_path], speaker_info, output)
    assert second['totals']['existing_verified'] == 1


def test_rejects_unexpected_members_and_bad_metadata():
  with tempfile.TemporaryDirectory() as directory:
    base = Path(directory)
    speaker_info = base / 'speaker.info'
    speaker_info.write_text('0124 F\n', encoding='utf-8')
    assert read_speaker_info(speaker_info) == {'S0124': 'F'}
    archive_path = base / 'bad.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
      _member(archive, '../BAC009S0124W0001.wav', b'unsafe')
    try:
      extract_wavs([archive_path], speaker_info, base / 'wav')
    except ValueError as error:
      assert 'Unexpected non-JSON member' in str(error)
    else:
      raise AssertionError('Unsafe member was not rejected.')
