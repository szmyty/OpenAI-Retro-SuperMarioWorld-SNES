import retro 
import numpy as np
##pip install opencv-python to install cv2
import cv2
import neat
import pickle

# Play this retro game at this level.
env = retro.make('SuperMarioWorld-Snes', 'DonutPlains1.state')

imgarray = []

def eval_genomes(genomes, config):
    
    for genome_id, genome in genomes:
        ob = env.reset()
        ac = env.action_space.sample() # action with generic sample

        iny, inx, inc = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        # Create a Recurrent Neural Network.
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        score = 0
        scoreTracker = 0
        coins = 0
        coinsTracker = 0
        yoshiCoins = 0
        yoshiCoinsTracker = 0
        xPosPrevious = 0
        yPosPrevious = 0
        checkpoint = False
        checkpointValue = 0
        endOfLevel = 0
        powerUps = 0
        powerUpsLast = 0
        jump = 0

        done = False

        while not done:
            env.render()
            frame += 1

            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            imgarray = ob.flatten()

            nnOutput = net.activate(imgarray)   
            
            ob, rew, done, info = env.step(nnOutput)        

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
            
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                     'config-feedforward')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

