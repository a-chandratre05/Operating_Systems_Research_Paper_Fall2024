import numpy as np
import matplotlib.pyplot as plt

# Parameters
time_steps = 20  # Number of time steps (e.g., seconds or minutes)
num_servers = 2   # Number of servers
spike_time = 15   # Time step at which the spike occurs
return_normal_time = 18 # Time step at which the spike ends
spike_magnitude = 200  # Magnitude of the load spike
threshold = 20   # Threshold value to start load balancing

# Simulate normal load (random) and a spike at a specific time
np.random.seed(42)  # For reproducibility
normal_load = np.random.randint(1, 10, size=time_steps)  # Random normal load
load_with_spike = normal_load.copy()

# Introduce a spike at the given time step
load_with_spike[spike_time:] += spike_magnitude  # Add spike after `spike_time`

# End spike at the given time step
load_with_spike[return_normal_time:] -= spike_magnitude

# Simulate load balancing that starts after the threshold is reached
def load_balance(load_data, num_servers, threshold, base_response_time, response_time_multiplier):
    # Create a matrix of loads across servers (rows = time steps, columns = servers)
    server_loads = np.zeros((len(load_data), num_servers))
    server_response_times = np.zeros((len(load_data), num_servers))

    for t in range(len(load_data)):
        load = load_data[t]

        if load > threshold:
            # If load exceeds the threshold, distribute it across servers
            remaining_load = load
            for server in range(num_servers):
                server_loads[t, server] = (remaining_load / num_servers) * 0.95
                remaining_load -= server_loads[t, server]
                
                server_response_times[t, server] = base_response_time + (server_loads[t, server] * response_time_multiplier)
        else:
            # If load is below threshold, assign to only one server (server 0)
            server_loads[t, 0] = load
            server_response_times[t, 0] = base_response_time + (load * response_time_multiplier)


    return server_loads, server_response_times 

# Perform load balancing with a threshold on the data with spike
balanced_loads, balanced_response_times = load_balance(load_with_spike, num_servers, threshold, base_response_time, response_time_multiplier)

# Plotting the results
time = np.arange(time_steps)

plt.figure(figsize=(10, 6))

# Plot load for each server
plt.subplot(2, 1, 1)  # Plot the load on servers

plt.plot(time, balanced_loads[:, 0], label=f"Server 1", marker='o')
plt.plot(time, balanced_loads[:, 1], label=f"Server 2", marker='s')


# Highlight the spike period
plt.axvline(x=spike_time, color='r', linestyle='--', label='Spike Start')
plt.axvline(x=return_normal_time, color='g', linestyle=':', label='Spike End')

# Mark the threshold
plt.axhline(y=threshold, color='grey', linestyle='-.', label='Threshold')

# Labels and title
plt.xlabel("Time")
plt.ylabel("Load")
plt.title("Load Balancing with Sudden Load Spike")
plt.legend(loc='upper left')
plt.grid(True)

# Plot response for each server
plt.subplot(2, 1, 2)  # Plot the response time on servers

plt.plot(time, balanced_response_times[:, 0], label=f"Server 1", marker='o')
plt.plot(time, balanced_response_times[:, 1], label=f"Server 2", marker='s')

# Highlight the spike period
plt.axvline(x=spike_time, color='r', linestyle='--', label='Spike Start')
plt.axvline(x=return_normal_time, color='g', linestyle=':', label='Spike End')

# Labels and title
plt.xlabel("Time")
plt.ylabel("Response Time (ms)")
plt.title("Server Response Times Under Load")
plt.legend(loc='upper left')

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()
