"""
Initialization of the ODE solver and its assiciated functions
"""

import numpy as np
import pandas as pd
import os
# import commands for finding optimal asteroid parameters
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt
from scipy import interpolate


class Planet():
    """
    The class called Planet is initialised with constants appropriate
    for the given target planet, including the atmospheric density profile
    and other constants
    """

    def __init__(self, atmos_func='exponential', atmos_filename=None,
                 Cd=1., Ch=0.1, Q=1e7, Cl=1e-3, alpha=0.3, Rp=6371e3,
                 g=9.81, H=8000., rho0=1.2, maxiter=50000):
        """
        Set up the initial parameters and constants for the target planet
        Parameters
        ----------
        atmos_func : string, optional
            Function which computes atmospheric density, rho, at altitude, z.
            Default is the exponential function rho = rho0 exp(-z/H).
            Options are 'exponential', 'tabular' and 'constant'
        atmos_filename : string, optional
            Name of the filename to use with the tabular atmos_func option
        Cd : float, optional
            The drag coefficient
        Ch : float, optional
            The heat transfer coefficient
        Q : float, optional
            The heat of ablation (J/kg)
        Cl : float, optional
            Lift coefficient
        alpha : float, optional
            Dispersion coefficient
        Rp : float, optional
            Planet radius (m)
        rho0 : float, optional
            Air density at zero altitude (kg/m^3)
        g : float, optional
            Surface gravity (m/s^2)
        H : float, optional
            Atmospheric scale height (m)
        """
        # Input constants
        self.Cd = Cd
        self.Ch = Ch
        self.Q = Q
        self.Cl = Cl
        self.alpha = alpha
        self.Rp = Rp
        self.g = g
        self.H = H
        self.rho0 = rho0
        self.maxiter = maxiter

        if np.isclose(g, 0.):  # setting analytical case paramaters
            self.Q = np.inf
            self.Rp = np.inf
            self.Cl = 0
            atmos_func = 'exponential'

        # define some constants
        self.atmos_func = atmos_func
        self.km_unit = 1000.0
        self.kt_unit = 4.184e12

        # define lambda functions
        # get energy by formula 0.5 * m * v ^ 2
        self.e_lambda = lambda m, v: 0.5 * m * v ** 2 / self.kt_unit
        # get burst energy by comparing which one is larger
        self.burst_energy = lambda b, r: b if b >= r else r
        # change the unit of altitude
        self.altitude_unit = lambda a: a / self.km_unit

        # set function to define atmoshperic density
        if atmos_func == 'exponential':
            self.rhoa = lambda z: self.rho0 * np.exp(-z/self.H)
        elif atmos_func == 'tabular':
            if atmos_filename == None:
                filename = os.path.join(os.path.dirname(__file__), '../data/AltitudeDensityTable.csv')
            else:
                filename = os.path.join(os.path.dirname(__file__), atmos_filename)
            df = pd.read_csv(filename, header=5, delimiter=" ")
            df = df.dropna(axis=1)
            df.columns = ['Altitude', 'rhoa', 'H']
            self.alt_tab = df['Altitude']
            self.rho_tab = df['rhoa']
            self.rhoa = lambda z: self.rho_tab[(self.alt_tab == np.abs(np.around(z, -1)))].values[0]

        elif atmos_func == 'constant':
            self.rhoa = lambda x: rho0
        else:
            raise NotImplementedError(
                "atmos_func must be 'exponential', 'tabular' or 'constant'")

    def multistep_rk4(self, y0, strength, A, density, ddt, steps=1):
        """
        Do multiple steps of the RK4 scheme.
        Parameters
        ----------
        y0 : np.array
            y0[:] = velocity, mass, angle, altitude, distance, radius
        strength : float
            Y - strength of asteroid
        A : function
            Function to evaluate the area from the radius
        density : float
            density of the meteor
        Output
        ------
        y0 : np.array
            ODE outputs, y0[:] = velocity, mass, angle, altitude, distance,
            radius
        """

        for _ in range(steps):
            k1 = ddt*self.f(y0, strength, A, density)
            k2 = ddt*self.f(y0 + 0.5*k1, strength, A, density)
            k3 = ddt*self.f(y0 + 0.5*k2, strength, A, density)
            k4 = ddt*self.f(y0 + k3, strength, A, density)
            y0 = y0 + (1./6.)*(k1 + 2*k2 + 2*k3 + k4)
        return y0

    def f(self, y0, strength, A, density):
        """
        Evaluate the coupled set of ODE's
        Parameters
        ----------
        y0 : np.array
            y0[:] = velocity, mass, angle, altitude, distance, radius
        strength : float
            Y - strength of asteroid
        A : function
            Function to evaluate the area from the radius
        density : float
            density of the meteor
        Output
        ------
        y1 : np.array
            ODE outputs, y0[:] = velocity, mass, angle, altitude, distance,
            radius
        """
        y1 = np.zeros(6)
        rho_a = self.rhoa(y0[3])
        vel = y0[0]

        # Velocity
        y1[0] = ((-(self.Cd * rho_a * A(y0[5]) * vel**2) /
                 (2 * y0[1])) + self.g * np.sin(y0[2]))

        # Mass
        y1[1] = -(self.Ch * rho_a * A(y0[5]) * vel**3)/(2 * self.Q) 

        # Angle
        y1[2] = (((self.g * np.cos(y0[2]))/vel) -
                 (self.Cl * rho_a * A(y0[5]) * vel) / (2*y0[1]) -
                 (vel * np.cos(y0[2])) / (self.Rp+y0[3]))

        # Altitude
        y1[3] = -abs(vel * np.sin(y0[2])) 

        # Distance
        y1[4] = abs(vel * np.cos(y0[2]) / (1 + (y0[3] / self.Rp)))

        # Radius
        if rho_a * vel**2 >= strength:
            y1[5] = np.sqrt(7./2. * self.alpha * rho_a/density) * vel
        else:
            y1[5] = 0

        return y1

    def solve_atmospheric_entry(
            self, radius, velocity, density, strength, angle,
            init_altitude=86e3, dt=0.05, radians=False):
        """
        Solve the system of differential equations for a given impact scenario
        Parameters
        ----------
        radius : float
            The radius of the asteroid in meters
        velocity : float
            The entery speed of the asteroid in meters/second
        density : float
            The density of the asteroid in kg/m^3
        strength : float
            The strength of the asteroid (i.e. the maximum pressure it can
            take before fragmenting) in N/m^2
        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians
        init_altitude : float, optional
            Initial altitude in m
        dt : float, optional
            The output timestep, in s
        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the dataframe will have the same units as the
            input
        Returns
        -------
        Result : DataFrame
            A pandas dataframe containing the solution to the system.
            Includes the following columns:
            'velocity', 'mass', 'angle', 'altitude',
            'distance', 'radius', 'time'
        """
        if np.isclose(self.g, 0.):
            strength = np.inf

        if self.atmos_func == 'tabular' and init_altitude > 86e3:
            raise IndexError('This altitude is out of range of the table')

        # Calculating area
        A = lambda radius: np.pi*y0[5]**2

        # Moving average of all variables
        med_fun = lambda x1, x0: x1 - 0.1 * (x1 - x0)

        # Initial conditions
        mass = (4./3.)*np.pi*(radius**3)*density

        # Convert into radians
        if not radians:
            angle = angle * (np.pi/180)

        # Initial conditions
        y0 = np.array(
            [velocity, mass, angle, init_altitude, 0.0, radius]
            )

        # Initialize data matrices
        y = np.copy(y0)
        t = np.array([0.])
        steps_count = 10
        ddt = dt/10
        counter = 0

        # Initialize booleans and the moving average
        breakup_bool = False
        steady_state = False
        first_run = True
        runmed = y0[[0, 1, 3, 4, 5]]
        final_counter = 0

        # Combat timeout error by setting max number of iterations
        # only do 200 steps after steady state
        while counter < 200 and final_counter < self.maxiter:  
            final_counter += 1
            # Implementing RK4
            y1 = self.multistep_rk4(
                y0, strength, A, density, ddt, steps=steps_count
                )

            # if we broke up during last step, redo with increased fidelity
            if y1[5] > radius and breakup_bool is False:
                breakup_bool = True
                y1 = self.multistep_rk4(
                    y0, strength, A, density, ddt/10, steps=steps_count*10
                    )

            # Assign data
            y0 = y1
            y = np.vstack((y, y0))
            t = np.append(t, t[-1] + dt)
            runmed = med_fun(runmed, y0[[0, 1, 3, 4, 5]])

            # Decrease fidelity after first run
            if first_run:
                ddt = dt
                steps_count = 1
                first_run = False

            # If we start breaking up, increase fidelity
            if breakup_bool:
                ddt = dt/10
                steps_count = 10

            # If we reach a steady state after breakup, start
            # counting and decrease fidelity
            if breakup_bool & steady_state:
                counter += 1
                ddt = dt
                steps_count = 1

            # if moving below ground, break
            if y0[3] <= 0:
                break

            # if all parameters have reached a steady state, set bool
            if np.allclose(
                 y0[[0, 1, 3, 4, 5]], runmed, rtol=1e-02, atol=1e-02
                 ) & breakup_bool:
                steady_state = True

        # Converting radians to degrees
        if not radians:
            y[:, 2] = y[:, 2] * (180/np.pi)

        return pd.DataFrame({'velocity': y[:, 0],
                             'mass': y[:, 1],
                             'angle': y[:, 2],
                             'altitude': y[:, 3],
                             'distance': y[:, 4],
                             'radius': y[:, 5],
                             'time': t[:]})

    def calculate_energy(self, result):
        """
        Function to calculate the kinetic energy lost per unit altitude in
        kilotons TNT per km, for a given solution.
        Parameters
        ----------
        result : DataFrame
            A pandas dataframe with columns for the velocity, mass, angle,
            altitude, horizontal distance and radius as a function of time
        Returns
        -------
        result : DataFrame
            Returns the dataframe with additional column ``dedz`` which is the
            kinetic energy lost per unit altitude
        """
        # Replace these lines with your code to add the dedz column to
        # the result DataFrame
        result = result.copy()

        # Create a new column to store dedz results
        dedzs = np.zeros(len(result.velocity))

        # change the pandas to numpy to get a fast speed
        result_numpy = result.to_numpy()

        # define the index of mass, velocity, altitude in one line
        v_index = 0
        m_index = 1
        a_index = 3

        for i in range(1, len(result.velocity)):
            # get the energy of this time point
            now_energy_line = result_numpy[i]
            now_energy = self.e_lambda(now_energy_line[m_index],
                                       now_energy_line[v_index])

            # get the energy of the previous time point
            pre_energy_line = result_numpy[i-1]
            previous_energy = self.e_lambda(pre_energy_line[m_index],
                                            pre_energy_line[v_index])

            # get the altitude of this time point
            now_altitude = self.altitude_unit(now_energy_line[a_index])

            # get the altitude of previous time point
            previous_altitude = self.altitude_unit(pre_energy_line[a_index])

            # get the average rate of kinetic energy lost in this timestep
            # which is the kinetic energy lost per unit altitude
            energy_difference = now_energy - previous_energy
            altitude_difference = now_altitude - previous_altitude
            dedzs[i] = abs(energy_difference / altitude_difference)

        result.insert(len(result.columns),
                      'dedz', dedzs)
        return result

    def analyse_outcome(self, result):
        """
        Inspect a pre-found solution to calculate the impact and airburst stats
        Parameters
        ----------
        result : DataFrame
            pandas dataframe with velocity, mass, angle, altitude, horizontal
            distance, radius and dedz as a function of time
        Returns
        -------
        outcome : Dict
            dictionary with details of the impact event, which should contain
            the key ``outcome`` (which should contain one of the
            following strings:
            ``Airburst``, ``Cratering`` or ``Airburst and cratering``),
            as well as
            the following keys:
            ``burst_peak_dedz``, ``burst_altitude``, ``burst_distance``,
            ``burst_energy``
        """
        # get the column where the burst happen
        burst_column_index = result['dedz'].idxmax()

        # get the burst line in the dataframe
        burst_line = result.iloc[burst_column_index]

        # get the burst peak dedz
        burst_peak_dedz = burst_line['dedz']

        # get the burst altitude by the burst column index
        burst_altitude = burst_line['altitude']

        # get the burst distance
        burst_distance = burst_line['distance']

        # get the burst velocity
        burst_velocity = burst_line['velocity']

        # get the burst mass
        burst_mass = burst_line['mass']

        # create a null dictionary
        outcome = {'outcome': 'Unknown',
                   'burst_peak_dedz': 0.,
                   'burst_altitude': 0.,
                   'burst_distance': 0.,
                   'burst_energy': 0.}

        # modify the null number to the result what we got previous
        outcome['burst_peak_dedz'] = burst_peak_dedz
        outcome['burst_altitude'] = burst_altitude
        outcome['burst_distance'] = burst_distance

        # get the break up point index
        after_breakup_points = result.radius.unique()

        if len(after_breakup_points) > 1:  # all events with breakup
            after_breakup_radius = after_breakup_points[1]
        else:  # cratering event without breakup
            after_breakup_radius = after_breakup_points[0]

        after_breakup_index_list = result[result.radius
                                          == after_breakup_radius].index
        after_breakup_index = after_breakup_index_list.tolist()[0] - 1

        # get the break up point values
        # get the break line of the dataframe
        break_up_line = result.iloc[after_breakup_index]
        break_up_velocity = break_up_line['velocity']
        break_up_mass = break_up_line['mass']
        # break_up_energy = self.get_energy(break_up_mass, break_up_velocity)
        break_up_energy = self.e_lambda(break_up_mass, break_up_velocity)

        # get the burst point energy
        # burst_point_energy = self.get_energy(burst_mass, burst_velocity)
        burst_point_energy = self.e_lambda(burst_mass, burst_velocity)

        # get the burst energy when altitude is enough (Airburst case)
        # but we need compare it with the residual energy in Airburst
        # and cratering and Cratering case
        burst_energy = burst_point_energy - break_up_energy

        # get residual energy which is the last line of the datafram
        # get the residual point line
        residual_line = result.iloc[-1]
        residual_mass = residual_line['mass']
        residual_velocity = residual_line['velocity']
        residual_energy = self.e_lambda(residual_mass, residual_velocity)

        # energy_diff = burst_peak_dedz - burst
        # judge different 3 types event
        if burst_altitude > 5:
            outcome['outcome'] = 'Airburst'
            outcome['burst_energy'] = abs(burst_energy)

        elif burst_altitude <= 5 and burst_altitude >= 0:
            outcome['outcome'] = 'Airburst and cratering'
            # this is the low altitude airbust
            # so we need to compare the larger of total kinetic energy loss or
            # the residual kinetic energy
            burst_energy = self.burst_energy(burst_energy, residual_energy)

        else:
            outcome['outcome'] = 'Cratering'
            # The same as Airburst and cratering
            burst_energy = self.burst_energy(burst_energy, residual_energy)

            # in ProjectDescription.ipynb, it says in Cratering case
            # the burst altitude should be 0
            outcome['burst_altitude'] = 0

            # the peak kinetic energy loss per unit height in kt per km
            # is the value at ground
            ground_line = result.iloc[-1]
            outcome['burst_peak_dedz'] = ground_line['dedz']

            # the distance should be the last one point's distance
            outcome['burst_distance'] = ground_line['distance']

        # set the burst energy
        outcome['burst_energy'] = abs(burst_energy)

        return outcome

    def minimax(self, y, sol):
        """
        Returns the maximum error in the given region
        Parameters
        ----------
        y : np.array
            Simulated values
        sol : np.array
            Actual values
        """
        return np.max(np.abs(y - sol))

    def find_minima(self, x):
        """
        Finding a global minima given an initial guess x
        """
        int_f = self.init_f()
        lw = [1] * 2
        up = [20] * 2
        res = dual_annealing(
            self.costfun, bounds=list(zip(lw, up)),
            maxfun=30, x0=x, args=[int_f])
        return res

    def costfun(self, params, int_f):
        """
        Assembly of the cost function for finding the optimal paramaters for
        curve fitting
        """
        result = self.solve_atmospheric_entry(radius=params[0], angle=18.3,
                                              strength=params[1] * 10 ** 6,
                                              density=3300,
                                              velocity=19.2e3,
                                              init_altitude=42.5e3,
                                              dt=0.1)
        # Calculate the kinetic energy lost per unit altitude and add it
        # as a column to the result dataframe
        result2 = self.calculate_energy(result)
        result2 = result2[['altitude', 'dedz']]
        # make sure only data points in the correct range make it through
        cond = result2['altitude'] < 42192
        cond = result2[cond * result2['altitude'] > 21620]  # x2
        xnew = cond['altitude']/1000  # make sure data is in km not m
        return self.minimax(int_f(xnew), cond['dedz'])

    def init_f(self):
        """
        Initializing minimization routine
        """
        filename = os.path.join(
            os.path.dirname(__file__), '../data/ChelyabinskEnergyAltitude.csv'
            )
        df = pd.read_csv(filename, delimiter=",")
        df.columns = ['altitude', 'energy']
        ff = interpolate.interp1d(df['altitude'], df['energy'])
        return ff

    def plot_minima(self, params):
        """
        Plot the power loss curve dependent on param compared with actual curve
        """
        int_f = self.init_f()
        result = self.solve_atmospheric_entry(radius=params[0], angle=18.3,
                                              strength=params[1] * 10 ** 6,
                                              density=3300,
                                              velocity=19.2e3,
                                              init_altitude=42.5e3,
                                              dt=0.1)
        # Calculate the kinetic energy lost per unit altitude and add it
        # as a column to the result dataframe
        result2 = self.calculate_energy(result)
        result2 = result2[['altitude', 'dedz']]
        # make sure only data points in the correct range make it through
        cond = result2['altitude'] < 42192
        cond = result2[cond * result2['altitude'] > 21620]  # x2
        xnew = cond['altitude'] / 1000  # make sure data is in km not m

        ynew = int_f(xnew)   # use interpolation function returned by interp1d
        fig = plt.figure(figsize=(15, 7))
        ax1 = plt.subplot(1, 1, 1)
        ax1.plot(xnew, cond['dedz'], 'o', label='Optimized values')
        ax1.plot(xnew, ynew, '-', label='Interpolation values of actual curve')
        ax1.set_xlabel('velocity [km/h]')
        ax1.set_ylabel('[kt km-1]')
        ax1.legend(loc='best')