![image](./img/static-logo-alt.png)

<p align=center>
   <i>Using PPO to beat snake.</i>
</p>
<br>
  
> [!NOTE]  
> Python 3.12 was used for this project



<br>
<br>


# 📖 INDEX:
 * 📌 [Project Overview](#-project-overview)
 * 🌐 [Model Structure](#-model-structure)
 * 🥇 [Reward Shaping](#-reward-shaping)
 * 👀 [Model Input](#-model-input)
 * 🎖️ [Resoults](#️-resoults)
 * 🚀 [Project Structure](#-project-structure)
 * 🤝 [Credits](#-credits)
 * 📄 [Licence](#-licence)

<br>
<br>
<br>
<br>



# 📌 Project Overview

This project was made with the goal of using [**PPO**](https://en.wikipedia.org/wiki/Proximal_policy_optimization) *(Proximal Policy Optimization)* to beat the game of [**snake**](https://en.wikipedia.org/wiki/Snake_(video_game_genre)).   
The main goal was to train a reinforcement learning agent capable of learning the game from scratch.  
Along the way, this project became a deep dive into PPO: understanding how it works, tuning hyperparameters, stabilizing training, and analyzing learning behavior through metrics. 

> Here is a gameplay demo of the final agent trained for 20M timesteps  
> (You can find the model in `/agent/pretrained_model.pth`)

<p align=center>
   <img src="./img/gameplays/game3.gif" alt='gameplay'>
</p>

<br>
<br>

# 🌐 Model Structure

The `ppo_agent.py` file contains 2 classes:
- The **PPOAgent** class, wich contains an implementation of the PPO algorithm
- The **FeedForwardNN** class, wich is the **ActorCritic** model

The **ActorCritic** model is made of a **CNN** with 3 convolutional layers that goes from 32 features to 64 and 2 fully connected layers of 32 neurons each
 
![image](./img/ActorCritic/ActorCritic-alt-2.svg)


<br>
<br>

# 🥇 Reward Shaping

The reward shaping is quite simple: **only +1 if the snake eats** food and when the snake dies, it simply starts a new game.  
This reward shaping might seem too sparse for PPO but out of all the rewards shaping I've tryied, it performed the best.

<br>
<br>

# 👀 Model Input

The agent receives an observation tensor of shape **(C, H, W)** directly from the environment.  
In this project the shape is (3, 10, 10):  
-	Channel 0 → Snake **head** position  
- Channel 1 → Snake **tail** positions  
- Channel 2 → **Food** position  
  
*Each channel is a binary grid (0 or 1) aligned with the game board.  
Before being passed to the network, the observation is batched to shape (N, 3, 10, 10) for PyTorch.*

![image](./img/observations-alt.png)

<br>
<br>

# 🎖️ Resoults

After training the model for 20M timesteps, here are the resoults:

> This is the reward graph, wich shows the model learning and getting more reward.  
> This is also the score graph since score and reward coincide.

![image](./img/graphs/reward-graph.png)
  
> This is the episode length graph.
> It represents the duration of the games during training.

![image](./img/graphs/ep-length-graph.png)

> Those are the configurations used during training.
 ```json
 {  
     "agent": {  
         "timestamps_per_batch": 4800,  
         "max_timestamps_per_episode": 1600,  
         "gamma": 0.95,  
         "n_updates_per_iteration": 5,  
         "clip": 0.2,  
         "lr": 0.0003  
     },  
     "env": {  
         "max_steps": 1000,  
         "obs_shape": [  
             3,  
             10,  
             10  
         ],  
         "action_shape": [ 4 ]  
     }  
 }
 ```


<br>
<br>

# 🚀 Project Structure  

<br>

**`train.py`**: the script where you can train the model.  
*(If you wonna make another training script, you can simply use `model.learn(timesteps)`)*  
Params:  
```py
--train-ts 20_000_000  # Number of timestamps the model will be trained for
```
```py
--ci 50_000  # Number of steps between checkpoint saves
```
```py
--vf 500  # Number of steps after wich the agent will be playing a live game, to see how it's doing
```

<br>

---

**`play.py`**: the script where you can test the trained model once training is completed.  
Params:  
```py
--path "agent\pretrained_model.pth"  # Path to the model file (file name included). It can be both absolute or relative to project's root folder
```
```py
--disable-gui  # No value needed, deactivates the gui on the environment (it will use the cli)
```

<br>

---

**`plot.py`**: the script where you can plot the data collected automatically from training.  
*(no params)*



<br>
<br>

# 🤝 Credits
Special thanks to [Eric Yang Yu](https://ericyangyu.github.io/) for the [PPO tutorial](https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8) and [Ettore](https://sa1g.github.io) for helping me with debugging.


<br>
<br>

# 📄 Licence
This project was released under [MIT License](https://github.com/paolomalgarin/snake-ppo/blob/main/LICENSE.txt).
