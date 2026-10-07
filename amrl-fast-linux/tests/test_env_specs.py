from amrl_fast.envs import get_environment_spec


def test_new_continuous_control_specs_are_registered():
    for name in ("Hopper-v2", "Walker2d-v2", "HalfCheetah-v2"):
        spec = get_environment_spec(name)
        assert spec.continuous_actions
        assert spec.candidates[0] == name
