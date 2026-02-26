from matplotlib import pyplot as plt
import json, os


if __name__ == '__main__':
    # Create paths
    training_logs_path = os.path.join('results', 'logs', 'training_logs.json')

    # Check file existance
    if not os.path.exists(training_logs_path):
        print("Logs not found, train the model with train.py before plotting.")

    # Load data
    with open(training_logs_path) as f:
        data = json.load(f)

    # Extract values
    batch_ns = []  # Batch numbers
    t_so_far = []  # Timesteps reached after the batch
    batch_steps = []  # Timesteps made in the batch
    avg_rewards = []  # Average batch reward
    max_rewards = []
    min_rewards = []
    avg_episode_lengths = []
    max_episode_lengths = []
    min_episode_lengths = []


    for elm in data:
        batch_ns.append(elm['batch'])
        t_so_far.append(elm['timesteps_so_far'])
        batch_steps.append(elm['batch_steps'])

        avg_rewards.append(elm['avg_reward'])
        max_rewards.append(elm['max_reward'])
        min_rewards.append(elm['min_reward'])

        avg_episode_lengths.append(elm['avg_episode_length'])
        max_episode_lengths.append(elm['max_episode_length'])
        min_episode_lengths.append(elm['min_episode_length'])


    # Display graphs
    colors = {
        'background': '#f0e5e1',
        'graph-bg': '#f9f5f3',
        'text': '#36241c',
        'reward-graph': {
            'line': '#fc7b28',
            'error': '#ffad33',
        },
        'len-graph': {
            'line': '#595959',
            'error': '#7f7f7f',
        },
    }

    plt.figure(facecolor=colors['background'], num='Training stats')

    # Rewards graph
    plt.subplot(2, 1, 1)
    plt.fill_between(t_so_far, min_rewards, max_rewards, color=colors['reward-graph']['error'], alpha=0.6)
    plt.plot(t_so_far, avg_rewards, color=colors['reward-graph']['line'])
    
    plt.xlabel('timesteps', color=colors['text'])
    plt.title('Rewards', color=colors['text'])

    plt.gca().set_facecolor(colors['graph-bg'])
    plt.gca().tick_params(colors=colors['text'])
    for spina in plt.gca().spines.values():
        spina.set_color(colors['text'])

    plt.grid(True, linestyle='--', alpha=0.7)

    # Episode lengths graph
    plt.subplot(2, 1, 2)
    plt.fill_between(t_so_far, min_episode_lengths, max_episode_lengths, color=colors['len-graph']['error'], alpha=0.6)
    plt.plot(t_so_far, avg_episode_lengths, color=colors['len-graph']['line'])
    
    plt.xlabel('timesteps', color=colors['text'])
    plt.title('Episode lengths', color=colors['text'])

    plt.gca().set_facecolor(colors['graph-bg'])
    plt.gca().tick_params(colors=colors['text'])
    for spina in plt.gca().spines.values():
        spina.set_color(colors['text'])
    
    plt.grid(True, linestyle='--', alpha=0.7)


    # Display graphs
    plt.tight_layout()
    plt.show()