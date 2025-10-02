
from iact_tools.models import calculate_theta2_classic
def test_theta2_not_none():
    assert calculate_theta2_classic(1.0, 500.0, 0.05, 0.2, 30.0) is not None
