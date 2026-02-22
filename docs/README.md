![image](./img/static-logo-alt.png)

<br>
<br>


# 📖 INDEX:
 * 📌 [Project Overview](#-project-overview)
 * 🌐 [Model Structure](#-model-structure)
 * 🥇 [Reward System](#-reward-system)
 * 👀 [Model Input](#-model-input)
 * 🚀 [Project Structure](#-project-structure)
    * 🎮 [Play](#-play)
    * 💪 [Train](#-train)
    * 📈 [Plot](#-plot)
    * 📊 [Debug Scripts](#-debug-scripts)
 * 🤝 [Credits](#-credits)
 * 📄 [Licence](#-licence)

<br>
<br>
<br>
<br>



# 📌 Project Overview

> [!NOTE]  
> Python 3.11.9 used 4 this project


This project was made to use **PPO** *(Proximal Policy Optimization)* to beat the game of [**snake**](https://en.wikipedia.org/wiki/Snake_(video_game_genre))

![image](./img/gameplays/game3.gif)
![image](./img/reward-graph.png)
![image](./img/ep-length-graph.png)

<br>
<br>

# 🌐 Model Structure

 *Le applicazioni front-end mandano le richieste all'API che è l'unico che può comunicare con il ML grazie ad un **HMAC***
 


<br>
<br>

# 🥇 Reward System

lorem ipsum dolor sit amet ...

<br>
<br>

# 👀 Model Input

lorem ipsum dolor sit amet ...
![image](./img/observations-alt.png)

<br>
<br>

# 🚀 Project Structure

lorem ipsum dolor sit amet ...

 - ## 🎮 Play
    lorem ipsum dolor sit amet ...

 - ## 💪 Train
    lorem ipsum dolor sit amet ...  
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

 - ## 📈 Plot
    lorem ipsum dolor sit amet ...

 - ## 📊 Debug Scripts
    lorem ipsum dolor sit amet ...

<br>
<br>

# 🤝 Thanks
SPECIAL THANKS TO:
 [Eric Yang Yu](https://ericyangyu.github.io/) for the PPO [tutorial](https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8) and [Ettore](https://sa1g.github.io) for helping me in general.


<br>
<br>

# 📄 Licence
This project was released under [MIT License](https://github.com/paolomalgarin/snake-ppo/blob/main/LICENSE.txt).
