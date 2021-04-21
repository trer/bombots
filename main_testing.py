from bombots.environment import Bombots
from templates.agent_NOP import NOPAgent
from templates.agent_rulebased import RuleBasedAgent
from templates.agent_that_does_not_suicide import BeatNopAgent
from templates.agent_testing import TestAgent
from templates.agent_testing_old import OLDTestAgent

# For Travis
import sys
if '--novid' in sys.argv: 
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"

env = Bombots(
    scale       = 64,                    # Size of game tiles (in pixels)
    framerate   = 10,                    # Frames per second, set this to 0 for unbounded framerate
    state_mode  = Bombots.STATE_DICT,    # So the state is returned as a dictionary
    verbose     = True,                  # Useful printing during execution
    render_mode = Bombots.RENDER_GFX_RGB # Change this to Bombots.NO_RENDER if you remove the render call
)

agents = [TestAgent(env), NOPAgent(env)]
# agents = [TestAgent(env), RuleBasedAgent(env)]
#agents = [TestAgent(env), OLDTestAgent(env)]
# agents = [BeatNopAgent(env), RuleBasedAgent(env)]

if '--test' not in sys.argv:
    states = env.reset()

    while True:
        states, rewards, done, info = env.step([agents[i].act(states[i]) for i in range(len(agents))])
        
        env.render() # Comment out this call to train faster
        
        if done: states = env.reset()