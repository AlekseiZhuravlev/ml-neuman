import colmap_helper
import logging

def test_read_cameras(caplog):
    caplog.set_level(logging.DEBUG)
    cameras, metas = colmap_helper.ColmapAsciiReader.read_cameras_and_meta('s')
    # print('test_read_cameras\n', metas[f'{100:05d}'])
    # print('test_read_cameras\n', metas[f'{101:05d}'])
    assert True


def test_read_captures(caplog):
    caplog.set_level(logging.DEBUG)
    result = colmap_helper.ColmapAsciiReader.read_captures('s', '0', '0', tgt_size=None)
    # print(result)
    assert True

def test_read_scene(caplog):
    caplog.set_level(logging.INFO)
    result = colmap_helper.ColmapAsciiReader.read_scene('s', '0')
    # print(result)
    assert True

