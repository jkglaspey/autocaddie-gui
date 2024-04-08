import json
import matplotlib.pyplot as plt
import math
import os

# Get the IMU data from the json file
def load_graphs(file_path):
    # Load the data from the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Extract the time, x, y, and z values from the data. w is unnecessary, as it exists to prevent gimbal lock
    times = [float(obj['time']) for obj in data]
    ws = [float(obj['w']) for obj in data]
    xs = [float(obj['x']) for obj in data]
    ys = [float(obj['y']) for obj in data]
    zs = [float(obj['z']) for obj in data]

    # Subtract the initial time from all time values to make it 0-indexed
    initial_time = times[0]
    times = [(t - initial_time) / 1000.0 for t in times]  # Convert to seconds

    return times, ws, xs, ys, zs

# Convert quaternion data to IMU data
def convert_quaternions_to_euler(ws, xs, ys, zs):
    pitches = []
    yaws = []
    rolls = []

    for w, x, y, z in zip(ws, xs, ys, zs):

        # Roll (x)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        rolls.append(math.degrees(math.atan2(t0, t1)))
    
        # Pitch (y)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitches.append(math.degrees(math.asin(t2)))
    
        # Yaw (z)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaws.append(math.degrees(math.atan2(t3, t4)))
    
    return pitches, yaws, rolls

# Plot everything
def convert_quaternions_to_graphs(times, xs, ys, zs, threshold, idx):
    
    # Colors
    green = '#00FF00'  # Bright green
    dark_green = '#006600'  # Dark green
    blue = '#0000FF'  # Bright blue
    dark_blue = '#000080'  # Dark blue
    orange = '#FF8C00'  # Orange
    dark_orange = '#BF6900'  # Dark orange
    red = '#FF0000'  # Red

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot the lines with threshold behavior and connect points
    x_points = []
    y_points = []
    z_points = []

    x_prev = xs[0]
    y_prev = ys[0]
    z_prev = zs[0]
    x_color_prev = dark_green
    y_color_prev = dark_blue
    z_color_prev = dark_orange

    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        x_color = red if abs(x - xs[0]) > threshold else dark_green
        y_color = red if abs(y - ys[0]) > threshold else dark_blue
        z_color = red if abs(z - zs[0]) > threshold else dark_orange

        if i != 0:
            if x_color != red and x_color_prev != red:
                ax.plot([times[i-1], times[i]], [x_prev, x], color=x_color_prev, linestyle='-')
            elif x_color == red or x_color_prev == red:
                ax.plot([times[i-1], times[i]], [x_prev, x], color='red', linestyle='--')
            if y_color != red and y_color_prev != red:
                ax.plot([times[i-1], times[i]], [y_prev, y], color=y_color_prev, linestyle='-')
            elif y_color == red or y_color_prev == red:
                ax.plot([times[i-1], times[i]], [y_prev, y], color='red', linestyle='--')
            if z_color != red and z_color_prev != red:
                ax.plot([times[i-1], times[i]], [z_prev, z], color=z_color_prev, linestyle='-')
            elif z_color == red or z_color_prev == red:
                ax.plot([times[i-1], times[i]], [z_prev, z], color='red', linestyle='--')

        x_prev = x
        y_prev = y
        z_prev = z
        x_color_prev = x_color
        y_color_prev = y_color
        z_color_prev = z_color

        x_point = ax.plot(times[i], x, color=x_color, linestyle='None', marker='o', markerfacecolor=x_color)
        y_point = ax.plot(times[i], y, color=y_color, linestyle='None', marker='o', markerfacecolor=y_color)
        z_point = ax.plot(times[i], z, color=z_color, linestyle='None', marker='o', markerfacecolor=z_color)

        x_points.extend(x_point)
        y_points.extend(y_point)
        z_points.extend(z_point)

    # Add error lines
    ax.axhline(y=xs[0], color=green, linestyle='--', linewidth=1, label='_nolegend_')
    ax.axhline(y=xs[0] + threshold, color=green, linestyle='-.', linewidth=1, label='_nolegend_')
    ax.axhline(y=xs[0] - threshold, color=green, linestyle='-.', linewidth=1, label='_nolegend_')

    ax.axhline(y=ys[0], color=blue, linestyle='--', linewidth=1, label='_nolegend_')
    ax.axhline(y=ys[0] + threshold, color=blue, linestyle='-.', linewidth=1, label='_nolegend_')
    ax.axhline(y=ys[0] - threshold, color=blue, linestyle='-.', linewidth=1, label='_nolegend_')

    ax.axhline(y=zs[0], color=orange, linestyle='--', linewidth=1, label='_nolegend_')
    ax.axhline(y=zs[0] + threshold, color=orange, linestyle='-.', linewidth=1, label='_nolegend_')
    ax.axhline(y=zs[0] - threshold, color=orange, linestyle='-.', linewidth=1, label='_nolegend_')

    # Set labels and title
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Normalized Component Value')
    ax.set_title(f'IMU #{idx} Preparatory Measurements')

    # Add a legend
    ax.legend(['X', 'Y', 'Z'])

    # Save the plot
    plot_filename = f"quaternion_graph_{idx}"
    plot_filename = os.path.join(r"data_processing\imu_data", plot_filename)
    plt.savefig(plot_filename)
    plt.clf()



# Plot roll information
def plot_roll(times, rolls, threshold, idx):
    # Colors
    green = '#00FF00'  # Bright green
    dark_green = '#006600'  # Dark green
    red = '#FF0000'  # Red

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the roll data
    for i, roll in enumerate(rolls):
        color = red if abs(roll - rolls[0]) > threshold else dark_green
        if i != 0:
            linestyle = '-' if color != red and color_prev != red else '--'
            ax.plot([times[i-1], times[i]], [roll_prev, roll], color=color_prev, linestyle=linestyle)
        roll_prev = roll
        color_prev = color
        ax.plot(times[i], roll, color=color, linestyle='None', marker='o', markerfacecolor=color)

    # Plot threshold lines
    ax.axhline(y=rolls[0], color=green, linestyle='--', linewidth=1, label='_nolegend_')
    ax.axhline(y=rolls[0] + threshold, color=green, linestyle='-.', linewidth=1, label='_nolegend_')
    ax.axhline(y=rolls[0] - threshold, color=green, linestyle='-.', linewidth=1, label='_nolegend_')

    # Set labels and title
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title(f'IMU #{idx}: Roll (X-axis)')

    # Save the plot
    plot_filename = f"roll_graph_{idx}"
    plot_filename = os.path.join(r"data_processing\imu_data", plot_filename)
    plt.savefig(plot_filename)
    plt.clf()

# Plot pitch information
def plot_pitch(times, pitches, threshold, idx):
    # Colors
    blue = '#0000FF'  # Bright blue
    dark_blue = '#000080'  # Dark blue
    red = '#FF0000'  # Red

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the pitch data
    for i, pitch in enumerate(pitches):
        color = red if abs(pitch - pitches[0]) > threshold else dark_blue
        if i != 0:
            linestyle = '-' if color != red and color_prev != red else '--'
            ax.plot([times[i-1], times[i]], [pitch_prev, pitch], color=color_prev, linestyle=linestyle)
        pitch_prev = pitch
        color_prev = color
        ax.plot(times[i], pitch, color=color, linestyle='None', marker='o', markerfacecolor=color)

    # Plot threshold lines
    ax.axhline(y=pitches[0], color=blue, linestyle='--', linewidth=1, label='_nolegend_')
    ax.axhline(y=pitches[0] + threshold, color=blue, linestyle='-.', linewidth=1, label='_nolegend_')
    ax.axhline(y=pitches[0] - threshold, color=blue, linestyle='-.', linewidth=1, label='_nolegend_')

    # Set labels and title
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title(f'IMU #{idx}: Pitch (Y-axis)')

    # Save the plot
    plot_filename = f"pitch_graph_{idx}"
    plot_filename = os.path.join(r"data_processing\imu_data", plot_filename)
    plt.savefig(plot_filename)
    plt.clf()

# Plot yaw information
def plot_yaw(times, yaws, threshold, idx):
    # Colors
    orange = '#FF8C00'  # Orange
    dark_orange = '#BF6900'  # Dark orange
    red = '#FF0000'  # Red

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the yaw data
    for i, yaw in enumerate(yaws):
        color = red if abs(yaw - yaws[0]) > threshold else dark_orange
        if i != 0:
            linestyle = '-' if color != red and color_prev != red else '--'
            ax.plot([times[i-1], times[i]], [yaw_prev, yaw], color=color_prev, linestyle=linestyle)
        yaw_prev = yaw
        color_prev = color
        ax.plot(times[i], yaw, color=color, linestyle='None', marker='o', markerfacecolor=color)

    # Plot threshold lines
    ax.axhline(y=yaws[0], color=orange, linestyle='--', linewidth=1, label='_nolegend_')
    ax.axhline(y=yaws[0] + threshold, color=orange, linestyle='-.', linewidth=1, label='_nolegend_')
    ax.axhline(y=yaws[0] - threshold, color=orange, linestyle='-.', linewidth=1, label='_nolegend_')

    # Set labels and title
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title(f'IMU #{idx}: Yaw (Z-axis)')

    # Save the plot
    plot_filename = f"yaw_graph_{idx}"
    plot_filename = os.path.join(r"data_processing\imu_data", plot_filename)
    plt.savefig(plot_filename)
    plt.clf()

# Wrapper function
def plot_pyr(times, rolls, pitches, yaws, threshold, idx):
    plot_roll(times, rolls, threshold, idx)
    plot_pitch(times, pitches, threshold, idx)
    plot_yaw(times, yaws, threshold, idx)


# Wrapper function
def process_imu_jsons(threshold_q, threshold_e):

    # Q1
    times, ws, xs, ys, zs = load_graphs(r'data_processing\imu_data\prep_data_q1.json')
    convert_quaternions_to_graphs(times, xs, ys, zs, threshold_q, 1)
    rolls, pitches, yaws = convert_quaternions_to_euler(ws, xs, ys, zs)
    plot_pyr(times, rolls, pitches, yaws, threshold_e, 1)
    

    # Q2
    times, ws, xs, ys, zs = load_graphs(r'data_processing\imu_data\prep_data_q2.json')
    convert_quaternions_to_graphs(times, xs, ys, zs, threshold_q, 2)
    rolls, pitches, yaws = convert_quaternions_to_euler(ws, xs, ys, zs)
    plot_pyr(times, rolls, pitches, yaws, threshold_e, 2)

if __name__ == "__main__":
    process_imu_jsons(0.1,5)
