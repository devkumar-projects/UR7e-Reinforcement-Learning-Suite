import unittest,numpy as np
from ur7e_env_v4 import fk as original_fk,HOME_CONFIG
from kinematics import fk
from ur7e_traj_env import UR7eTrajEnv
from policy import Policy
class ReleaseChecks(unittest.TestCase):
 def test_portable_fk_matches_source(self):
  rng=np.random.default_rng(17011)
  for q in rng.uniform(-np.pi,np.pi,(50,6)):np.testing.assert_allclose(fk(q),original_fk(q),atol=2e-7,rtol=1e-6)
 def test_original_success_flag_is_not_strict_completion(self):
  traj=np.tile(original_fk(HOME_CONFIG)+[.04,0,0],(60,1));e=UR7eTrajEnv(fixed_trajectory=traj);e.reset();e._phase=1;e._stagnation=150
  _,_,done,_,info=e.step(np.zeros(6));self.assertTrue(done);self.assertTrue(info['completed']);self.assertEqual(e._cursor,0);self.assertGreater(np.linalg.norm(fk(e._angles)-traj[-1]),.025)
 def test_portable_actor_validates_inputs(self):
  p=Policy();a=p.predict(np.zeros(29,dtype=np.float32));self.assertEqual(a.shape,(6,));self.assertTrue(np.isfinite(a).all());self.assertTrue((np.abs(a)<=1).all())
  with self.assertRaises(ValueError):p.predict(np.zeros(28))
  with self.assertRaises(ValueError):p.predict(np.full(29,np.nan))
if __name__=='__main__':unittest.main()
