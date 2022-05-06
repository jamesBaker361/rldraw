from gym.spaces import Discrete
import numpy as np

class DiscretePrime(Discrete):
    def sample(self) -> int:
        return int(np.random.randint(self.start,self.n+1))