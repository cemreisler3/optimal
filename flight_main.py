import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

class FlightModel:
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        
    def __init__(self, dt=0.1):
        #constants
        self.g = 9.81
        self.m = 68000 
        self.dt = dt
        #initial states
        self.x=0
        self.y=0
        self.h=8000
        self.v = 220  #m/s
        self.psi = 0  #initial heading angle(rad)
        
        #aerodynamics
        self.cd0 = 0.025452
        self.k = 0.035815 
        self.wingarea = 124.65
        
        #fuel consumption
        self.cfcr = 0.92958 
        self.cf1 = 0.70057
        self.cf2 = 1068.1
        
        #engine thrust
        self.tcr = 0.95
        self.tc1 = 146590
        self.tc2 = 53872
        self.tc3 = 3.0453e-11
        
        #wind 
        self.cx = [-21.151, 10.0039, 1.1081, -0.5239, -0.1297, -0.006, 0.0073, 0.0066, -0.0001]
        self.cy = [-65.3035, 17.6148, 1.0855, -0.7001, -0.5508, -0.003, 0.0241, 0.0064, -0.000227]
        
        #plot variables
        self.x_history = []
        self.y_history = []
        self.h_history = []
        self.v_history = []
        self.m_history = []
        self.throttle_history = []
        self.thrust_history = []
        self.bank_history= []
        self.fpath_history = [] #flight path angle
        
    def rho(self, h):
        rho0 = 1.225
        exponent = -2.2257e-5 * h
        return rho0 * (1 - exponent)**4.2586
    
    def lift_coeff(self,v,h,mu):
        rho=self.rho(h)
        cl = (2* self.m * self.g) / (rho * v**2 * self.wingarea * np.cos(mu))
        return cl
    
    def drag_coeff(self,v,h,mu):
        cl=self.lift_coeff(v,h,mu)
        cd = self.cd0 + (self.k*cl**2)
        return cd
    
    def maxthrust(self,h):
        factor = (1 - (3.28 * h / self.tc2) + self.tc3 * (3.28 * h)**2)
        max_thrust = self.tcr * self.tc1 * factor
        return max_thrust
    
    def fuel_eff(self,v):
        return self.cf1 / 6000 * (1 + 1.943 * v / self.cf2)
    
    def fuel_consumption(self,v,h, delta):
        max_thrust = self.maxthrust(h)
        eta = self.fuel_eff(v)
        fuel_usage = delta*max_thrust*eta*self.cfcr
        return fuel_usage
    
    #def wind_speed_x(self, x, y):
        cx = self.cx
        return (cx[0] + cx[1] * x + cx[2] * y + cx[3] * x * y + 
                cx[4] * x**2 + cx[5] * y**2 + cx[6] * x**2 * y + 
                cx[7] * x * y**2 + cx[8] * x**2 * y**2)

    #def wind_speed_y(self, x, y):
        cy = self.cy
        return (cy[0] + cy[1] * x + cy[2] * y + cy[3] * x * y + 
                cy[4] * x**2 + cy[5] * y**2 + cy[6] * x**2 * y + 
                cy[7] * x * y**2 + cy[8] * x**2 * y**2)

    def update_state(self, delta, gamma , mu): #delta, gamma and mu are control inputs
        rho = self.rho(self.h)
        max_thrust = self.maxthrust(self.h)
        thrust = delta * max_thrust
        drag = (self.cd0 + (self.k * ((2 * self.m * self.g) / (rho * self.v**2 * self.wingarea * np.cos(mu)))**2)) * 0.5 * rho * self.v**2 * self.wingarea
        
        # wind_x = self.wind_speed_x(self.x, self.y)
        # wind_y = self.wind_speed_y(self.x, self.y)
        
        #updating velocities and positions (also heading angle)
        self.v += self.dt * (thrust-drag)/self.m - self.g * np.sin(gamma)
        self.psi += self.dt * ((2 * self.m * self.g) / (rho * self.v * self.wingarea * np.cos(mu)) * np.sin(mu) / (self.m * np.cos(gamma)))
        self.x += self.dt * ( self.v * np.cos(self.psi) * np.cos(gamma))
        self.y += self.dt * (self.v * np.sin(self.psi) * np.cos(gamma))
        #self.x += self.dt * ( self.v * np.cos(self.psi) * np.cos(gamma) + wind_x)
        #self.y += self.dt * (self.v * np.sin(self.psi) * np.cos(gamma) + wind_y)
        self.h += self.dt * self.v * np.sin(gamma)
        #updating mass
        self.m -= self.fuel_consumption(self.h, self.v, delta)
        
        self.x_history.append(self.x)
        self.y_history.append(self.y)
        self.h_history.append(self.h)
        self.v_history.append(self.v)
        self.m_history.append(self.m)
        self.throttle_history.append(delta)
        self.thrust_history.append(thrust)
        self.bank_history.append(mu)
        self.fpath_history.append(gamma) 
        
class FlightEnv(gym.Env):
    def __init__(self, target_x,target_y,target_h, model):
        super(FlightEnv, self).__init__()
        self.action_space = spaces.Box(low=np.array([0, -np.pi, -np.pi/2]), high=np.array([1, np.pi, np.pi/2]), dtype=np.float32)  # control variables
        self.observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, 0, 0, -np.pi, 0]), high=np.array([np.inf, np.inf, np.inf, np.inf, np.pi, np.inf]), dtype=np.float32)
        
        self.target = np.array([target_x, target_y, target_h])
        self.flight_model = model
        self.total_fuel_used = 0
        
    def reset(self):
        self.flight_model.x = 0
        self.flight_model.y = 0
        self.flight_model.h = 8000
        self.flight_model.v = 220
        self.flight_model.psi = 0
        self.flight_model.m = 68000
        self.total_fuel_used = 0
        return np.array([self.flight_model.x, self.flight_model.y, self.flight_model.h, self.flight_model.v, self.flight_model.psi, self.flight_model.m])
    
    def step(self, action):
        delta, mu, gamma = action
        
        self.flight_model.update_state(delta,gamma,mu)
        self.total_fuel_used += self.flight_model.fuel_consumption(self.flight_model.h, self.flight_model.v, delta)
        
        new_state = np.array([self.flight_model.x, self.flight_model.y, self.flight_model.h, self.flight_model.v, self.flight_model.psi, self.flight_model.m])
        
        position_error = np.sum((new_state[:3] - self.target)**2)
        reward = - (position_error + self.total_fuel_used)
        
        done = self.flight_model.h <= 0 or self.flight_model.v <= 0 or self.flight_model.m <= 0
        return new_state, reward, done, {}
    
    def render(self, mode='human'):
        pass
        
flight_model = FlightModel(dt=0.1) 
env = FlightEnv(target_x=10000, target_y=10000, target_h=9000, model=flight_model)    
model = PPO("MlpPolicy", env, verbose=1)

#training
model.learn(total_timesteps=10000)
#saving
model.save("ppo_flightmodel")
model = PPO.load("ppo_flightmodel", env=env)
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
        
def plot_flight_data(model):
    t = [i * model.dt for i in range(len(model.x_history))]
    fig, axs = plt.subplots(8, 1, figsize=(10, 20))
    fig.tight_layout(pad=3.0)

    # X-Y Position
    if model.x_history and model.y_history:
        axs[0].plot(model.x_history, model.y_history, label='X-Y Trajectory')
        axs[0].set_xlabel('X Position (meters)')
        axs[0].set_ylabel('Y Position (meters)')
        axs[0].legend()
        axs[0].grid(True)

    # Altitude-Time
    axs[1].plot(t, model.h_history, label='Altitude')
    axs[1].set_xlabel('Time (seconds)')
    axs[1].set_ylabel('Altitude (meters)')
    axs[1].legend()
    axs[1].grid(True)

    # Airspeed-Time
    axs[2].plot(t, model.v_history, label='True Airspeed (Vtas)')
    axs[2].set_xlabel('Time (seconds)')
    axs[2].set_ylabel('True Airspeed (m/s)')
    axs[2].legend()
    axs[2].grid(True)

    # Mass-Time
    axs[3].plot(t, model.m_history, label='Mass')
    axs[3].set_xlabel('Time (seconds)')
    axs[3].set_ylabel('Mass (kg)')
    axs[3].legend()
    axs[3].grid(True)

    # Thrust-Time
    axs[4].plot(t, model.thrust_history, label='Thrust')
    axs[4].set_xlabel('Time (seconds)')
    axs[4].set_ylabel('Thrust (N)')
    axs[4].legend()
    axs[4].grid(True)

    # Throttle-Time
    axs[5].plot(t, model.throttle_history, label='Throttle')
    axs[5].set_xlabel('Time (seconds)')
    axs[5].set_ylabel('Throttle (%)')
    axs[5].legend()
    axs[5].grid(True)

    # Bank Angle-Time
    axs[6].plot(t, model.bank_history, label='Bank Angle')
    axs[6].set_xlabel('Time (seconds)')
    axs[6].set_ylabel('Bank Angle (radians)')
    axs[6].legend()
    axs[6].grid(True)

    # Flight Path Angle-Time
    axs[7].plot(t, model.fpath_history, label='Flight Path Angle')
    axs[7].set_xlabel('Time (seconds)')
    axs[7].set_ylabel('Flight Path Angle (radians)')
    axs[7].legend()
    axs[7].grid(True)

    plt.show()

plot_flight_data(flight_model)

