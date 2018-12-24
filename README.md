# Open AI Gym with Super Mario World for SNES

This project contains my implementation of the NEAT-Python algorithm to use a recurrent neural network that enables an AI-controlled Mario to train itself to complete levels in "Super Mario World" on SNES. I used the gym-retro-integration program to create my own variables from the game's RAM values and then used those variables to reward/penalize the AI. If the AI completes a level, the neural network is saved as the winner. 

To setup OpenAI Gym Retro, follow the instructions found on their github page: <https://github.com/openai/retro>

To install the neat-python package, follow the instructions found on their website: <https://neat-python.readthedocs.io/en/latest/installation.html>

## Trying this project on your machine

To run this project, you will need a Super Mario World SNES ROM. You will need to figure out how to get that on your own. 

Once you do so, you will need to import that following the instruction on the Open AI Gym Retro github page.

Next, you need to change the data.json file to the one that I have provided. For me, the path to this file is `C:\Users\Alan\Anaconda3\envs\OpenAI-Retro-Demos\Lib\site-packages\retro\data\stable\SuperMarioWorld-Snes\`

For this project, I created a conda environment called "OpenAI-Retro-Demos", which is why it is stored at that path.

To create a conda environment (optional), you need to install Anaconda (found at <https://www.anaconda.com/download/>) and then open an Anaconda prompt and type in this command:
`conda create --name myenv`

Once you do that, you need to download the config-feedforward file and SuperMarioWorldAI-NEAT.py file into the same directory as each other. Then, run the program by typing this command:
`python SuperMarioWorld-Snes.py`

If that doesn't work, you may need to place the files into the same directory as the NEAT library (not sure on this).

## data.json

The data.json file contains all variables that represent specific values that are stored in RAM during the gameplay. These values are found by using the Open AI Retro Integration User Interface. This interface allows you to track specific values in RAM until you determine what the value stands for.

![](gym-integration-UI.gif)

For example, to find the x position of the player character, you can track the player by searching for all values that have not changed yet, then move the player right and track all of the values have increased. Repeat this until only a few values remain. Then, move the player left and track all values that have decreased. Only a few values in RAM remain. You can then store this value in RAM as a variable and export it to the data.json file. I did this for many variables that I found while player Super Mario World.

`{
  "info": {
    "checkpoint": {
      "address": 5070,
      "type": "|i1"
    },
    "coins": {
      "address": 8261055,
      "type": "|u1"
    },
    "endOfLevel": {
      "address": 8259846,
      "type": "|i1"
    },
    "lives": {
      "address": 8261399,
      "type": "|u1"
    },
    "powerups": {
      "address": 25,
      "type": "|i1"
    },
    "score": {
      "address": 8261428,
      "type": "<u4"
    },
    "x": {
      "address": 148,
      "type": "<u2"
    },
    "dead": {
      "address": 8257688,
      "type": "<u4"
    },
    "y": {
      "address": 114,
      "type": "<u4"
    },
    "jump": {
      "address": 8257747,
      "type": "<u4"
    },
    "yoshiCoins": {
      "address": 8262690,
      "type": "<u4"
    }
  }
}`

To use the Integration User Interface, it is easiest if you have Linux or macOS. Follow the instructions on the Open AI Gym Retro github. This is how you can create your own variables.

## SuperMarioWorldAI-NEAT.py

This project uses the NEAT algorithm to create a recurrent neural network that is used to train the character in the game to play the levels.

The code that rewards and penalizes the AI player was my made contribution to this code.

```python
	    score = info['score']
            coins = info['coins']
            yoshiCoins = info['yoshiCoins']
            dead = info['dead']
            xPos = info['x']
            yPos = info['y']
            jump = info['jump']
            checkpointValue = info['checkpoint']
            endOfLevel = info['endOfLevel']
            powerUps = info['powerups']

            # Add to fitness score if mario gains points on his score.
            if score > 0:
                if score > scoreTracker:
                    fitness_current = (score * 10)
                    scoreTracker = score
            
            # Add to fitness score if mario gets more coins.
            if coins > 0:
                if coins > coinsTracker:
                    fitness_current += (coins - coinsTracker)
                    coinsTracker = coins
        
            # Add to fitness score if marioe gets more yoshi coins.
            if yoshiCoins > 0:
                if yoshiCoins > yoshiCoinsTracker:
                    fitness_current += (yoshiCoins - yoshiCoinsTracker) * 10
                    yoshiCoinsTracker = yoshiCoins

            # As mario moves right, reward him slightly.
            if xPos > xPosPrevious:
                if jump > 0:
                    fitness_current += 10
                fitness_current += (xPos / 100)
                xPosPrevious = xPos
                counter = 0
            # If mario is standing still or going backwards, penalize him slightly.
            else: 
                counter += 1
                fitness_current -= 0.1                     
            
            # Award mario slightly for going up higher in the y position (y pos is inverted).
            if yPos < yPosPrevious:
                fitness_current += 10
                yPosPrevious = yPos
            elif yPos < yPosPrevious:
                yPosPrevious = yPos

            # If mario loses a powerup, punish him 1000 points.
            if powerUps == 0:
                if powerUpsLast == 1:
                    fitness_current -= 500
                    print("Lost Upgrade")
            # If powerups is 1, mario got a mushroom...reward him for keeping it.
            elif powerUps == 1:
                if powerUpsLast == 1 or powerUpsLast == 0:
                    fitness_current += 0.025       
                elif powerUpsLast == 2: 
                    fitness_current -= 500
                    print("Lost Upgrade")
            # If powerups is 2, mario got a cape feather...reward him for keeping it.
            elif powerUps == 2:
                fitness_current += 0.05
                
            powerUpsLast = powerUps

            # If mario doesn't get any rewards for 1000 frames or move forward, then he finishes.
            #if fitness_current > current_max_fitness: 
                #current_max_fitness = fitness_current
                #counter = 0
            #else:
                #counter += 1
                                  
            # If mario reaches the checkpoint (located at around xpos == 2425) then give him a huge bonus.           
            if checkpointValue == 1 and checkpoint == False:
                fitness_current += 20000
                checkpoint = True
           
            # If mario reaches the end of the level, award him automatic winner.
            if endOfLevel == 1:
                fitness_current += 1000000
                done = True

            # If mario is standing still or going backwards for 1000 frames, end his try.
            if counter == 1000:
                fitness_current -= 125
                done = True                

            # If mario dies, dead becomes 0, so when it is 0, penalize him and move on.
            if dead == 0:
                fitness_current -= 100
                done = True 

            if done == True:
                print(genome_id, fitness_current)

            genome.fitness = fitness_current
```

This can obviously be added to and tweaked to improve the AI, of which I am still doing.

The code that I wrote was inspired by the tutorials presented by Lucas Thompson found here:
<https://www.youtube.com/channel/UCLA_tAh0hX9bjl6DfCe9OLw>

![](OpenAI-Retro-SuperMarioWorld.gif)

## config-feedforward

This file is the file that tweaks how the NEAT algorithm works and these can be adjusted for particular needs and better results for particular situations.

To find out more about how this works, check out the description page found here: <https://neat-python.readthedocs.io/en/latest/config_file.html>
