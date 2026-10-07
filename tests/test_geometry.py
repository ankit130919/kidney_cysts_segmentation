import numpy as np

from cysts.common.geometry import max_diameter_mm, sphericity


def _ball(r=6, shape=(32, 32, 32)):
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    c = np.array(shape) // 2
    return ((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2) <= r * r


def test_diameter_of_a_sphere_is_about_its_diameter():
    m = _ball(r=6)
    d = max_diameter_mm(m, (1.0, 1.0, 1.0))
    assert 11.0 <= d <= 13.5, d


def test_anisotropic_spacing_is_honoured():
    m = _ball(r=6)
    assert max_diameter_mm(m, (1.0, 1.0, 3.0)) > max_diameter_mm(m, (1.0, 1.0, 1.0))


def test_sphericity_is_near_one_for_a_ball_and_lower_for_a_rod():
    assert sphericity(_ball(r=7), (1, 1, 1)) > 0.85
    rod = np.zeros((40, 10, 10), bool); rod[:, 4:6, 4:6] = True
    assert sphericity(rod, (1, 1, 1)) < 0.5


def test_single_voxel_does_not_crash():
    m = np.zeros((8, 8, 8), bool); m[4, 4, 4] = True
    assert max_diameter_mm(m, (1, 1, 1)) == 0.0
