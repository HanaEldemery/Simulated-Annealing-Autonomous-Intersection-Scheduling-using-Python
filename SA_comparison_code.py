import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np

def visualize_intersection_graphical(cars, x0, start_times, finish_times, makespan):
    """
    Creates an animated graphical visualization of the intersection
    """
    # Create the sequence
    lane_count = {'N': 0, 'E': 0, 'S': 0, 'W': 0}
    sequence = []
    for lane in x0:
        lane_count[lane] += 1
        car_index = lane_count[lane]
        found = False
        for car in cars:
            if car['lane'] == lane and car['i'] == car_index:
                sequence.append(car)
                found = True
                break
        # If not found (e.g., single car case), just append the first car with that lane
        if not found:
            for car in cars:
                if car['lane'] == lane:
                    sequence.append(car)
                    break

    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 8))
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+100+0")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw static intersection
    draw_intersection_base(ax)
    
    # Car colors for each lane
    colors = {'N': 'blue', 'S': 'red', 'E': 'green', 'W': 'orange'}
    
    # Initialize car objects
    car_patches = []
    car_texts = []
    for i, car in enumerate(sequence):
        color = colors[car['lane']]
        car_patch = patches.Circle((0, 0), 0.5, fc=color, ec='black', linewidth=2, alpha=0.8)
        ax.add_patch(car_patch)
        car_patches.append(car_patch)
        
        text = ax.text(0, 0, f"{car['lane']}_{car['dir']}", ha='center', va='center', 
                      fontsize=8, fontweight='bold', color='white')
        car_texts.append(text)
    
    # Status text
    time_text = ax.text(0, 13, '', ha='center', fontsize=14, fontweight='bold')
    status_text = ax.text(0, -13, '', ha='center', fontsize=10)
    
    max_time = int(makespan) + 1
    
    def get_car_position(lane, direction, progress):
        """
        Calculate car position based on lane, direction, and progress (0-1)
        progress: 0 = entering, 1 = exiting
        """
        # Starting positions (before intersection)
        starts = {
            'N': (0, 2),
            'S': (0, -2),
            'E': (2, 0),
            'W': (-2, 0)
        }
        
        # Ending positions (after intersection) based on direction
        ends = {
            ('N', 'R'): (-12, 0),   # North left to West
            ('N', 'S'): (0, -12),   # North straight to South
            ('N', 'L'): (12, 0),    # North right to East
            ('S', 'R'): (12, 0),    # South left to East
            ('S', 'S'): (0, 12),    # South straight to North
            ('S', 'L'): (-12, 0),   # South right to West
            ('E', 'R'): (0, 12),    # East left to North
            ('E', 'S'): (-12, 0),   # East straight to West
            ('E', 'L'): (0, -12),   # East right to South
            ('W', 'R'): (0, -12),   # West left to South
            ('W', 'S'): (12, 0),    # West straight to East
            ('W', 'L'): (0, 12)     # West right to North
        }
        
        start = starts[lane]
        end = ends[(lane, direction)]
        
        # For turns, create curved path
        if direction == 'L' or direction == 'R':
            # Create arc for turn
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            
            # Add curve to the path
            if progress < 0.5:
                t = progress * 2
                x = start[0] * (1 - t) + mid_x * t
                y = start[1] * (1 - t) + mid_y * t
            else:
                t = (progress - 0.5) * 2
                x = mid_x * (1 - t) + end[0] * t
                y = mid_y * (1 - t) + end[1] * t
        else:
            # Straight path
            x = start[0] * (1 - progress) + end[0] * progress
            y = start[1] * (1 - progress) + end[1] * progress
        
        return x, y
    
    def animate(frame):
        t = frame * 0.45  # Each frame = 0.45 time units
        
        if t > max_time:
            return car_patches + car_texts + [time_text, status_text]
        
        time_text.set_text(f'Time: {t:.1f} / {makespan}')
        
        waiting = 0
        moving = 0
        done = 0
        
        for i, car in enumerate(sequence):
            start = start_times[i]
            finish = finish_times[i]
            
            if t < start:
                # Waiting - hide car
                car_patches[i].set_visible(False)
                car_texts[i].set_visible(False)
                waiting += 1
            elif start <= t <= finish:
            #elif start <= t < finish:
                # Moving through intersection
                car_patches[i].set_visible(True)
                car_texts[i].set_visible(True)
                moving += 1
                
                # Calculate progress through intersection (0 to 1)
                progress = (t - start) / (finish - start)
                x, y = get_car_position(car['lane'], car['dir'], progress)
                
                car_patches[i].center = (x, y)
                car_texts[i].set_position((x, y))
            else:
                # Done - hide car
                car_patches[i].set_visible(False)
                car_texts[i].set_visible(False)
                done += 1
        
        status_text.set_text(f'Waiting: {waiting} | Moving: {moving} | Done: {done}')
        
        return car_patches + car_texts + [time_text, status_text]
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=int(max_time * 2) + 10, 
                         interval=200, blit=True, repeat=False)
    
    plt.title('Intersection Traffic Simulation', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def draw_intersection_base(ax):
    """Draw the static intersection roads"""
    road_width = 4
    road_color = '#404040'
    line_color = 'yellow'
    
    # Horizontal road (East-West)
    ax.add_patch(patches.Rectangle((-15, -road_width/2), 30, road_width, 
                                   fc=road_color, ec='none'))
    
    # Vertical road (North-South)
    ax.add_patch(patches.Rectangle((-road_width/2, -15), road_width, 30, 
                                   fc=road_color, ec='none'))
    
    # Center lines
    ax.plot([-15, -road_width/2], [0, 0], 'y--', linewidth=1, alpha=0.5)
    ax.plot([road_width/2, 15], [0, 0], 'y--', linewidth=1, alpha=0.5)
    ax.plot([0, 0], [-15, -road_width/2], 'y--', linewidth=1, alpha=0.5)
    ax.plot([0, 0], [road_width/2, 15], 'y--', linewidth=1, alpha=0.5)
    
    # Intersection box
    ax.add_patch(patches.Rectangle((-road_width/2, -road_width/2), 
                                   road_width, road_width, 
                                   fc='#505050', ec='white', linewidth=2))
    
    # Labels
    ax.text(0, 14, 'NORTH', ha='center', fontsize=12, fontweight='bold', color='white')
    ax.text(0, -14, 'SOUTH', ha='center', fontsize=12, fontweight='bold', color='white')
    ax.text(14, 0, 'EAST', ha='center', fontsize=12, fontweight='bold', color='white')
    ax.text(-14, 0, 'WEST', ha='center', fontsize=12, fontweight='bold', color='white')
    
    # Legend
    legend_y = 11
    ax.text(-11, legend_y, '●', color='blue', fontsize=20)
    ax.text(-10, legend_y, 'North', fontsize=9, va='center')
    ax.text(-6, legend_y, '●', color='red', fontsize=20)
    ax.text(-5, legend_y, 'South', fontsize=9, va='center')
    ax.text(-1, legend_y, '●', color='green', fontsize=20)
    ax.text(0, legend_y, 'East', fontsize=9, va='center')
    ax.text(4, legend_y, '●', color='orange', fontsize=20)
    ax.text(5, legend_y, 'West', fontsize=9, va='center')

CONFLICT_PAIRS = {
    ('N_L', 'N_L'), ('N_S', 'N_S'), ('N_R', 'N_R'),
    ('E_L', 'E_L'), ('E_S', 'E_S'), ('E_R', 'E_R'),
    ('S_L', 'S_L'), ('S_S', 'S_S'), ('S_R', 'S_R'),
    ('W_L', 'W_L'), ('W_S', 'W_S'), ('W_R', 'W_R'),

    ('S_R', 'W_S'), ('S_R', 'N_L'),
    ('S_S', 'E_R'), ('S_S', 'E_S'), ('S_S', 'E_L'), ('S_S', 'N_L'), ('S_S', 'W_S'), ('S_S', 'W_L'),
    ('S_L', 'W_L'), ('S_L', 'W_S'), ('S_L', 'N_R'), ('S_L', 'N_L'), ('S_L', 'N_S'), ('S_L', 'E_S'), ('S_L', 'E_L'),
    ('N_R', 'E_S'), ('N_R', 'S_L'),
    ('N_S', 'W_S'), ('N_S', 'W_R'), ('N_S', 'W_L'), ('N_S', 'S_L'), ('N_S', 'E_S'), ('N_S', 'E_L'),
    ('N_L', 'E_L'), ('N_L', 'E_S'), ('N_L', 'S_R'), ('N_L', 'S_L'), ('N_L', 'S_S'), ('N_L', 'W_S'), ('N_L', 'W_L'),
    ('E_R', 'S_S'), ('E_R', 'W_L'),
    ('E_S', 'S_S'), ('E_S', 'S_L'), ('E_S', 'W_L'), ('E_S', 'N_S'), ('E_S', 'N_L'), ('E_S', 'N_R'),
    ('E_L', 'N_L'), ('E_L', 'N_S'), ('E_L', 'W_R'), ('E_L', 'W_L'),('E_L', 'W_S'), ('E_L', 'S_S'), ('E_L', 'S_L'),
    ('W_R', 'N_S'), ('W_R', 'E_L'),
    ('W_S', 'N_S'), ('W_S', 'N_L'), ('W_S', 'E_L'), ('W_S', 'S_S'), ('W_S', 'S_L'), ('W_S', 'S_R'),
    ('W_L', 'S_L'), ('W_L', 'S_S'), ('W_L', 'E_R'), ('W_L', 'E_L'),('W_L', 'E_S'), ('W_L', 'N_S'), ('W_L', 'N_L'),
}

def random_swap(arr):
    n = len(arr)
    if n < 2:
        return arr  
    while True:
        i, j = random.sample(range(n), 2)

        if arr[i] != arr[j]:
            arr[i], arr[j] = arr[j], arr[i]
            break  
    return arr

def calculate_makespan_and_wait(x0, cars, conflict_pairs):
    start_times = []
    finish_times = []
    lane_count = {'N': 0, 'E': 0, 'S': 0, 'W': 0}
    sequence = []
    for lane in x0:
        lane_count[lane] += 1
        car_index = lane_count[lane]
        for car in cars:
            if car['lane'] == lane and car['i'] == car_index:
                sequence.append(car)
                break
    for i, car_i in enumerate(sequence):
        start_time = 0
        car_i_tag = f"{car_i['lane']}_{car_i['dir']}"
        for j in range(i):
            car_j = sequence[j]
            car_j_tag = f"{car_j['lane']}_{car_j['dir']}"
            if (car_i_tag, car_j_tag) in conflict_pairs or (car_j_tag, car_i_tag) in conflict_pairs:
                start_time = max(start_time, finish_times[j])
        start_times.append(start_time)
        finish_times.append(start_time + car_i['tc'])
    makespan = max(finish_times)
    total_waiting_time = sum(start_times)
    return makespan, total_waiting_time, start_times, finish_times

def plot_sa_convergence(iterations, makespan_history, waittime_history, fitness_history, best_makespan_history, best_waittime_history):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('SA Algorithm: Objective Functions Evolution', fontsize=16, fontweight='bold')
    
    # Plot 1: Makespan evolution
    ax1.plot(iterations, makespan_history, 'b-', alpha=0.7, linewidth=1, label='Current Makespan')
    ax1.plot(iterations, best_makespan_history, 'r-', linewidth=2, label='Best Makespan')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Makespan (time units)')
    ax1.set_title('Makespan Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Wait time evolution
    ax2.plot(iterations, waittime_history, 'g-', alpha=0.7, linewidth=1, label='Current Wait Time')
    ax2.plot(iterations, best_waittime_history, 'orange', linewidth=2, label='Best Wait Time')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Total Wait Time (time units)')
    ax2.set_title('Wait Time Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def SA(cars):
    x0 = [] 
    t0 = 500
    ti = 500
    tMin= 400
    b = 2.5 
    r = 0.2
    if len(cars) == 1:
        print("The car in lane: ", cars[0]['lane'], " will start moving at t = ", 0, " and finish at t = ", cars[0]['tc'])
        print("This will have a makespan of: ", cars[0]['tc'], " and a total wait time of: ", 0)  
        visualize_intersection_graphical([cars[0]], [cars[0]['lane']], [0], [cars[0]['tc']], cars[0]['tc'])
        return
    for i in range(len(cars)):
        x0.append(f"{cars[i]['lane']}")
    flag = False
    for i in range(len(x0)):
        if i < len(x0)-1:
            if cars[i]['lane'] != cars[i+1]['lane']:
                flag = True
    if flag == False:
        makespanx0, waitTimex0, startTimesx0, finishTimesx0 = calculate_makespan_and_wait(x0, cars, CONFLICT_PAIRS)
        print("START TIMES: ", startTimesx0)
        for i in range(len(x0)):
            print("The car in lane: ", x0[i], " will start moving at t = ", startTimesx0[i], " and finish at t = ", finishTimesx0[i])
        print("This will have a makespan of: ", makespanx0, " and a total wait time of: ", waitTimex0)
        visualize_intersection_graphical(cars, x0, startTimesx0, finishTimesx0, makespanx0)
        return 
    # reset the i so that the temperature doesn't decrease by the no of cars at the second iteration
    i = 1
    
    # Initialize best solution with first evaluation
    makespanx0, waitTimex0, startTimesx0, finishTimesx0 = calculate_makespan_and_wait(x0, cars, CONFLICT_PAIRS)
    best_fitness = 0.5 * makespanx0 + 0.5 * waitTimex0
    best_x = x0.copy()
    best_makespan = makespanx0
    best_wait = waitTimex0
    best_start = startTimesx0.copy()
    best_finish = finishTimesx0.copy()
    
    # Initialize tracking lists for plotting objective functions
    iterations = []
    makespan_history = []
    waittime_history = []
    fitness_history = []
    best_makespan_history = []
    best_waittime_history = []
    best_fitness_history = []
    
    # Main SA Loop
    while ti >= tMin:
        makespanx0, waitTimex0, startTimesx0, finishTimesx0 = calculate_makespan_and_wait(x0, cars, CONFLICT_PAIRS)
        print(f"Iteration {i}: Temperature = {ti:.2f}, Current Solution - Makespan: {makespanx0}, Wait Time: {waitTimex0}, Fitness: {0.5 * makespanx0 + 0.5 * waitTimex0:.2f}")
        
        # Store current iteration data for plotting
        iterations.append(i)
        makespan_history.append(makespanx0)
        waittime_history.append(waitTimex0)
        fitness_history.append(0.5 * makespanx0 + 0.5 * waitTimex0)
        best_fitness_history.append(best_fitness)
        best_makespan_history.append(best_makespan)
        best_waittime_history.append(best_wait)
        
        x1 = random_swap(x0.copy())
        makespanx1, waitTimex1, startTimesx1, finishTimesx1 = calculate_makespan_and_wait(x1, cars, CONFLICT_PAIRS)

        f0 = 0.5 * makespanx0 + 0.5 * waitTimex0
        f1 = 0.5 * makespanx1 + 0.5 * waitTimex1
        
        # Check if new solution (f1) is better than best so far
        if f1 < best_fitness:
            best_fitness = f1
            best_x = x1.copy()
            best_makespan = makespanx1
            best_wait = waitTimex1
            best_start = startTimesx1.copy()
            best_finish = finishTimesx1.copy()
            print(f"  New Best Fitness: {best_fitness:.2f}")
        
        # Check if incoming solution is better or worse
        if f1 > f0:
            deltaF = f1 - f0
            fraction = -deltaF / ti
            p = math.exp(fraction)
            r = random.uniform(0, 1)
            print(f"  -> Neighbor is worse: Makespan: {makespanx1}, Wait Time: {waitTimex1}, ΔF: {deltaF:.2f}, P(accept): {p:.4f}, Random: {r:.4f}")
            if r < p:
                x0 = x1.copy()
                makespanx0 = makespanx1
                waitTimex0 = waitTimex1
                startTimesx0 = startTimesx1.copy()
                finishTimesx0 = finishTimesx1.copy()
                print("  -> Accepted worse solution!")
            else:
                print("  -> Rejected worse solution")
        else:
            x0 = x1.copy()
            makespanx0 = makespanx1
            waitTimex0 = waitTimex1
            startTimesx0 = startTimesx1.copy()
            finishTimesx0 = finishTimesx1.copy()
            print(f"  -> Neighbor is better: Makespan: {makespanx1}, Wait Time: {waitTimex1} -> accepted!")
        
        ti = t0 - b * i
        i += 1
    
    print(f"\n=== SA COMPLETED ===")
    print(f"Final temperature: {ti + b:.2f}")
    print(f"Total iterations: {i-1}")
    
    # Plot objective functions evolution
    # plot_sa_convergence(iterations, makespan_history, waittime_history, fitness_history, best_makespan_history, best_waittime_history)
    

    print("==============================SA ALGORITHM SOLUTION==============================")
    for i in range(len(x0)):
        print("The car in lane: ", x0[i], " will start moving at t = ", startTimesx0[i], " and finish at t = ", finishTimesx0[i])
    print("This will have a makespan of: ", makespanx0, " and a total wait time of: ", waitTimex0)
    # visualize_intersection_graphical(cars, x0, startTimesx0, finishTimesx0, makespanx0)

    print(f"Optimal solution:")
    print("==============================BEST SOLUTION==============================")
    for i in range(len(best_x)):
        print("The car in lane: ", best_x[i], " will start moving at t = ", best_start[i], " and finish at t = ", best_finish[i])
    print("This will have a makespan of: ", best_makespan, " and a total wait time of: ", best_wait)
    # visualize_intersection_graphical(cars, best_x, best_start, best_finish, best_makespan)
    print(f"Best fitness: {best_fitness}")
    print("="*50)
    return {
    'best_solution': best_x,
    'best_fitness': best_fitness,
    'makespan': best_makespan,
    'wait_time': best_wait,
    'iterations': i-1,
    'best_fitness_history': best_fitness_history
    }, iterations, best_fitness_history



if __name__ == "__main__":
    cars = [
        {'lane': 'S', 'dir': 'L', 'tc': 5, 'i': 2},
        {'lane': 'N', 'dir': 'S', 'tc': 3, 'i': 2},
        {'lane': 'S', 'dir': 'R', 'tc': 2, 'i': 1},
        {'lane': 'N', 'dir': 'L', 'tc': 5, 'i': 1},
        {'lane': 'E', 'dir': 'S', 'tc': 6, 'i': 1},
        {'lane': 'W', 'dir': 'L', 'tc': 4, 'i': 1}
    ]
    print(SA(cars))
    # cars2 = [
    #     {'lane': 'S', 'dir': 'L', 'tc': 5, 'i': 3},
    #     {'lane': 'S', 'dir': 'S', 'tc': 3, 'i': 2},
    #     {'lane': 'S', 'dir': 'R', 'tc': 2, 'i': 1},
    #     {'lane': 'S', 'dir': 'L', 'tc': 6, 'i': 4},
    # ]
    # SA(cars2)
    # cars3 = [
    #     {'lane': 'S', 'dir': 'L', 'tc': 5, 'i': 1}
    # ]
    # SA(cars3)
    # cars4 = [
    #     {'lane': 'N', 'dir': 'R', 'tc': 5, 'i': 2},
    #     {'lane': 'N', 'dir': 'L', 'tc': 4, 'i': 1},
    #     {'lane': 'N', 'dir': 'S', 'tc': 3, 'i': 3},
    #     {'lane': 'S', 'dir': 'R', 'tc': 5, 'i': 2},
    #     {'lane': 'S', 'dir': 'L', 'tc': 4, 'i': 1},
    #     {'lane': 'S', 'dir': 'S', 'tc': 3, 'i': 3},
    #     {'lane': 'W', 'dir': 'R', 'tc': 5, 'i': 2},
    #     {'lane': 'W', 'dir': 'L', 'tc': 4, 'i': 1},
    #     {'lane': 'W', 'dir': 'S', 'tc': 3, 'i': 3},
    #     {'lane': 'E', 'dir': 'R', 'tc': 5, 'i': 2},
    #     {'lane': 'E', 'dir': 'L', 'tc': 4, 'i': 1},
    #     {'lane': 'E', 'dir': 'S', 'tc': 3, 'i': 3},    
    # ]
    # SA(cars4)
    # cars5 = [
    #     {'lane': 'N', 'dir': 'S', 'tc': 5, 'i': 4},
    #     {'lane': 'N', 'dir': 'S', 'tc': 4, 'i': 1},
    #     {'lane': 'N', 'dir': 'S', 'tc': 3, 'i': 3},
    #     {'lane': 'N', 'dir': 'S', 'tc': 3, 'i': 2}
    # ]
    # SA(cars5)