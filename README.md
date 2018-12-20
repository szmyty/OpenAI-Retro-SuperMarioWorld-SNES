-># Open AI Gym with Super Mario World for SNES<-

->This project contains my implementation of the NEAT-Python algorithm to use a recurrent neural network that enables an AI-controlled Mario to train itself to complete levels in "Super Mario World" on SNES. I used the gym-retro-integration program to create my own variables from the game's RAM values and then used those variables to reward/penalize the AI. If the AI complete's a level, the neural network is saved as the winner. <-

To setup OpenAI Gym Retro, follow the instructions found on their github page: <https://github.com/openai/retro>

To install the neat-python package, follow the instructions found on their website: <https://neat-python.readthedocs.io/en/latest/installation.html>

->## data.json<-

->The data.json file contains all variables that represent specific values that are stored in RAM during the gameplay. These values are found by using the Open AI Retro Integration User Interface. This interface allows you to track specific values in RAM until you determine what the value stands for.<-

->For example, to find the x position of the player character, you can track the player by searching for all values that have not changed yet, then move the player right and track all of the values have increased. Repeat this until only a few values remain. Then, move the player left and track all values that have decreased. Only a few values in RAM remain. You can then store this value in RAM as a variable and export it to the data.json file. I did this for many variables that I found while player Super Mario World.<-

'{
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
    "deathDetection": {
      "address": 8257688,
      "type": "<u4"
    },
    "yoshiCoins": {
      "address": 8262690,
      "type": "<u4"
    }
  }
}
'


![](OpenAI-Retro-SuperMarioWorld.gif)