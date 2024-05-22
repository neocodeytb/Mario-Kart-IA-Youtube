from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight,DirectionalLight,WindowProperties,Filename,LPoint3f,CollisionTraverser,CollisionHandlerQueue,CollisionPolygon,CollisionNode,CollisionSegment
from math import sin,cos
from direct.gui.DirectGui import *
from utils import degToRad
from panda3d.core import Point3,GeomVertexFormat,NodePath,Texture,TextureStage,CardMaker,CollisionBox,BitMask32
import neat
import os
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from playsound import playsound
import pygame
import threading

class Player():
    def __init__(self,x,y,z,model,name,ai,cshow):
        self.x = x
        self.y = y
        self.z = z
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.angle = 0
        self.model = model
        self.ia = ai

        self.tx = 0
        self.ty = 0
        self.tz = 0

        self.boost = 0
        self.v = 60
        self.w = 100
        self.static_count = 0

        self.name = f"Joueur{name}"
        self.score = 0

        # EN DEV :
        self.nbchampi = 3 #Champignons : Boost Vitesse
        self.drift = False #Dérapage

        self.onRace = True

        self.dangerLeft = 0
        self.dangerRight = 0
        self.roadLeft = 0
        self.roadRight = 0
        self.front = 0
        self.frontDanger = 0
        self.fd2 = 0
        self.dl2 = 0
        self.dr2 = 0
        self.rrs = 0
        self.rls = 0

        self.c = []
        self.cshow = cshow

class Game(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.sim = True
        self.initBackground()
        self.initColliders()
        self.importModels()
        self.initCamera()
        self.initLights()
        self.initControls()
        self.initMap()
        self.startMusic()
        self.setupWindowProperties()


       

        if self.sim:
            self.initNeat()
            self.runNeat()
        else:
            self.initControlledGame()
            self.playControlledGame()


        


    def setupWindowProperties(self):
        wp = WindowProperties()
        wp.setTitle("Mariokart-AI")
        wp.setIconFilename(Filename("assets/logo.ico"))
        self.win.requestProperties(wp)



    def startMusic(self):
        music_thread = threading.Thread(target=self.playMusic)
        music_thread.start()

    def playMusic(self):
        pygame.mixer.init()
        pygame.mixer.music.load(r'assets\music.mp3')
        pygame.mixer.music.play(-1)  # Boucle indéfiniment


    def initControlledGame(self):

        self.spawnPoint1 = (-451.135, -71.7493, 165.561)
        self.spawnPoint2 = (-76.135, -47.7493, 157.561)
        self.spawnPoint3 = (134, 89, 74)
        self.spawnPoint4 = (99, -203, 283)
        self.spawnPoint5 = (-176, -351, 170)
        self.currentSpawnPoint = self.spawnPoint1

        self.players = []

        self.tickmax = 1000.0
        self.tick = 0.0
        self.idx = 0

        node, cshow = self.createPlayerNode(self.currentSpawnPoint[0], self.currentSpawnPoint[1],self.currentSpawnPoint[2], 0)
        self.players.append(Player(self.currentSpawnPoint[0], self.currentSpawnPoint[1], self.currentSpawnPoint[2], node, 0, None,cshow))
        self.cameraStick = True

    def playControlledGame(self):
        self.taskMgr.add(self.controlledGameStep, "GameStep")
        self.taskMgr.add(self.verifFinGame, "VerifFinGame")
        self.taskMgr.add(self.updateCamera, "UpdateCamera")

    def controlledGameStep(self,task):
        dt = globalClock.getDt()
        self.manageCollisions(dt)
        self.moovePlayer(dt)
        return task.cont

    def moovePlayer(self,dt):

        dx, dy, dz = 0, 0, 0
        player = self.players[0]

        player.tx = player.x
        player.ty = player.y
        player.tz = player.z

        if self.keyMap["forward"]:  # Avancer
            dy += dt * player.v * sin(degToRad(player.model.getH() + 90))
            dx += dt * player.v * cos(degToRad(player.model.getH() + 90))

        # if actions_mask[1]:
        #     dy -= dt * player.v * sin(degToRad(player.model.getH() + 90))
        #     dx -= dt * player.v * cos(degToRad(player.model.getH() + 90))

        if self.keyMap["left"]:
            player.angle += player.w * dt
            player.model.setH(player.model.getH() + player.w * dt)
            self.camera.setH(player.model.getH())

        if self.keyMap["right"]:
            player.angle -= player.w * dt
            player.model.setH(player.model.getH() - player.w * dt)
            self.camera.setH(player.model.getH())

        dy += dt * player.boost * sin(degToRad(player.model.getH() + 90))
        dx += dt * player.boost * cos(degToRad(player.model.getH() + 90))
        player.boost = max(0, player.boost - 20 * dt)

        player.model.setPos(
            player.model.getX() + dx * 0.5,
            player.model.getY() + dy * 0.5,
            player.model.getZ() + dz * 0.5,
        )

        player.x += dx * 0.5
        player.y += dy * 0.5
        player.z += dz * 0.5

    def initNeat(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, "config.txt")
        self.config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,neat.DefaultStagnation, config_path)
        self.p = neat.Population(self.config)
        self.p.add_reporter(neat.StdOutReporter(True))
        self.reporter = neat.StatisticsReporter()
        self.p.add_reporter(self.reporter)

        self.players = []
        self.best_fitnesses = []
        self.mean_fitnesses = []

        self.tickmax = 1000.0
        self.tick = 0.0
        self.idx = 0

        self.taskMgr.stop()
        self.taskMgr.add(self.gameStep,"GameStep")
        self.taskMgr.add(self.verifFinGame, "VerifFinGame")
        self.taskMgr.add(self.updateCamera,"UpdateCamera")

        self.spawnPoint1 = (-451.135, -71.7493, 165.561)
        self.spawnPoint2 = (-76.135, -47.7493, 157.561)
        self.spawnPoint3 = (134,89,74)
        self.spawnPoint4 = (99,-203,283)
        self.spawnPoint5 = (-176,-351,170)

        self.currentSpawnPoint = self.spawnPoint1
        self.parametricExperiment = False

    def gameStep(self,task):
        dt = globalClock.getDt()
        self.manageCollisions(dt)
        self.moovePlayers(dt)
        return task.cont

    def runNeat(self):
        winner = self.p.run(self.playRace, 10000)
        print(f"Meilleur génome : {winner}")
        with open('best_model.pkl', 'wb') as f:
            pickle.dump(winner, f)
        sys.exit()

    def playRace(self,genomes,config):
        self.tick = 0.0
        self.finGame = False
        self.players = []
        self.idx = 0

        for i,g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)
            node,cshow = self.createPlayerNode(self.currentSpawnPoint[0],self.currentSpawnPoint[1], self.currentSpawnPoint[2],i)
            self.players.append(Player(self.currentSpawnPoint[0],self.currentSpawnPoint[1], self.currentSpawnPoint[2],node,i,net,cshow))
            g.fitness = 0

        if self.currentSpawnPoint[0] == self.spawnPoint1[0]:
            for player in self.players:
                player.score = 0

        if self.currentSpawnPoint[0] == self.spawnPoint2[0]:
            for player in self.players:
                player.score = 25

        if self.currentSpawnPoint[0] == self.spawnPoint3[0]:
            for player in self.players:
                player.score = 31

        if self.currentSpawnPoint[0] == self.spawnPoint4[0]:
            for player in self.players:
                player.score = 39

        if self.currentSpawnPoint[0] == self.spawnPoint5[0]:
            for player in self.players:
                player.score = 54

        if self.cameraStick:
            self.camera.setHpr(0,-20,0)

        self.taskMgr.run()

        idx = -1
        for _, g in genomes:
            idx += 1
            if idx < len(self.players):
                g.fitness = self.players[idx].score

        best_fitness = max([g.fitness for _, g in genomes])
        mean_fitnesses = np.mean(np.array([g.fitness for _, g in genomes]))

        if self.parametricExperiment:
            if best_fitness < 25:
                self.currentSpawnPoint = self.spawnPoint1
            if best_fitness >= 25 and best_fitness < 31:
                self.currentSpawnPoint = self.spawnPoint2
            if best_fitness >= 31 and best_fitness < 39:
                self.currentSpawnPoint = self.spawnPoint3
            if best_fitness >= 39 and best_fitness < 54:
                self.currentSpawnPoint = self.spawnPoint4
            if best_fitness >= 54:
                self.currentSpawnPoint = self.spawnPoint5
            if best_fitness == 68:
                self.currentSpawnPoint = self.spawnPoint1

        self.best_fitnesses.append(best_fitness)
        self.mean_fitnesses.append(mean_fitnesses)

        for player in self.players:
            if player.onRace:
                player.model.removeNode()
            for c in player.c:
                c.removeNode()
            del(player)


    def verifFinGame(self,task):
        self.finGame = self.verif()
        if self.finGame:
            self.taskMgr.stop()
        return task.cont

    def verif(self):
        if self.tick > self.tickmax:
            return True
        for player in self.players:
            if player.onRace:
                return False
        return True

    def manageCollisions(self,dt):
        for player in self.players:
            if player.onRace:
                player.vz -= 40*dt
                player.static_count += dt

                player.dangerLeft = 0
                player.dangerRight = 0
                player.roadLeft = 0
                player.roadRight = 0
                player.front = 0
                player.frontDanger = 0
                player.fd2 = 0
                player.dr2 = 0
                player.dl2 = 0
                player.rrs = 0
                player.rls = 0

        traversers = (self.traverserWalls,self.traverserScores,self.traverserMap,self.traverserBoosts,self.traverserWins,self.traverserInclinesP,self.traverserInclinesR,self.traverserJumpPads,self.traverserFly)
        for traverser in traversers:
            traverser.traverse(self.render)
            for idx in range(self.handler.getNumEntries()):
                entry = self.handler.getEntry(idx)
                intoNodePath = entry.getIntoNodePath()
                fromNodePath = entry.getFromNodePath()
                name_collided = intoNodePath.getName()
                name_collider = fromNodePath.getName()
                #print(name_collided,name_collider)

                if len(name_collider) > 11 and len(name_collided) > 11:

                    if (name_collided[:8] == "mesh_mdl" or name_collider[:8] == "mesh_mdl") and (name_collider[:17] == "JoueurRoadLeftSpe" or name_collided[:17] == "JoueurRoadLeftSpe"):
                        for player in self.players:
                            if player.name == name_collided[:6] + name_collided[17:] or player.name == name_collider[:6] + name_collider[17:] and player.onRace:
                                player.rls = 1

                    if (name_collided[:8] == "mesh_mdl" or name_collider[:8] == "mesh_mdl") and (name_collider[:18] == "JoueurRoadRightSpe" or name_collided[:18] == "JoueurRoadRightSpe"):
                        for player in self.players:
                            if player.name == name_collided[:6] + name_collided[18:] or player.name == name_collider[:6] + name_collider[18:] and player.onRace:
                                player.rrs = 1

                    if (name_collided[:8] == "mesh_mdl" or name_collider[:8] == "mesh_mdl") and (name_collider[:16] == "JoueurDangerLeft" or name_collided[:16] == "JoueurDangerLeft"):
                        for player in self.players:
                            if player.name == name_collided[:6] + name_collided[16:] or player.name == name_collider[:6] + name_collider[16:] and player.onRace:
                                player.dangerLeft = 1

                    if (name_collided[:8] == "mesh_mdl" or name_collider[:8] == "mesh_mdl") and (name_collider[:17] == "JoueurDangerLeft2" or name_collided[:17] == "JoueurDangerLeft2"):
                        for player in self.players:
                            if player.name == name_collided[:6] + name_collided[17:] or player.name == name_collider[:6] + name_collider[17:] and player.onRace:
                                player.dl2 = 1

                    if (name_collided[:8] == "mesh_mdl" or name_collider[:8] == "mesh_mdl") and (name_collider[:18] == "JoueurFrontDanger2" or name_collided[:18] == "JoueurFrontDanger2"):
                        for player in self.players:
                            if player.name == name_collided[:6] + name_collided[18:] or player.name == name_collider[:6] + name_collider[18:] and player.onRace:
                                player.fd2 = 1

                    if (name_collided[:8] == "mesh_mdl" or name_collider[:8] == "mesh_mdl") and (name_collider[:17] == "JoueurFrontDanger" or name_collided[:17] == "JoueurFrontDanger"):
                        for player in self.players:
                            if player.name == name_collided[:6] + name_collided[17:] or player.name == name_collider[:6] + name_collider[17:] and player.onRace:
                                player.frontDanger = 1

                    if (name_collided[:8] == "mesh_mdl" or name_collider[:8] == "mesh_mdl") and (name_collider[:11] == "JoueurFront" or name_collided[:11] == "JoueurFront"):
                        for player in self.players:
                            if player.name == name_collided[:6] + name_collided[11:] or player.name == name_collider[:6] + name_collider[11:] and player.onRace:
                                player.front = 1

                    if (name_collided[:8] == "mesh_mdl" or name_collider[:8] == "mesh_mdl") and (name_collider[:17] == "JoueurDangerRight" or name_collided[:17] == "JoueurDangerRight"):
                        for player in self.players:
                            if player.name == name_collided[:6] + name_collided[17:] or player.name == name_collider[:6] + name_collider[17:] and player.onRace:
                                player.dangerRight = 1

                    if (name_collided[:8] == "mesh_mdl" or name_collider[:8] == "mesh_mdl") and (name_collider[:18] == "JoueurDangerRight2" or name_collided[:18] == "JoueurDangerRight2"):
                        for player in self.players:
                            if player.name == name_collided[:6] + name_collided[18:] or player.name == name_collider[:6] + name_collider[18:] and player.onRace:
                                player.dr2 = 1

                    if (name_collided[:8] == "mesh_mdl" or name_collider[:8] == "mesh_mdl") and (name_collider[:14] == "JoueurRoadLeft" or name_collided[:14] == "JoueurRoadLeft"):
                        for player in self.players:
                            if player.name == name_collided[:6] + name_collided[14:] or player.name == name_collider[:6] + name_collider[14:] and player.onRace:
                                player.roadLeft = 1
                    if (name_collided[:8] == "mesh_mdl" or name_collider[:8] == "mesh_mdl") and (name_collider[:15] == "JoueurRoadRight" or name_collided[:15] == "JoueurRoadRight"):
                        for player in self.players:
                            if player.name == name_collided[:6] + name_collided[15:] or player.name == name_collider[:6] + name_collider[15:] and player.onRace:
                                player.roadRight = 1

                if (name_collided[:8] == "mesh_mdl" or name_collider[:8] == "mesh_mdl") and (name_collider[:6] == "Joueur" or name_collided[:6] == "Joueur"):
                    for player in self.players:
                        if player.name == name_collided or player.name == name_collider and player.onRace:
                            player.model.setZ(player.model.getZ() + 2*dt)
                            player.z += 2*dt
                            player.vz = max(0,player.vz)

                if (name_collided[:8] == "mesh_mdl" or name_collider[:8] == "mesh_mdl") and (name_collider[:9] == "JoueurFly" or name_collided[:9] == "JoueurFly"):
                    for player in self.players:
                        if player.name[6:] == name_collided[9:] or player.name[6:] == name_collider[9:] and player.onRace:
                            player.vz = max(0,player.vz)

                if (name_collided == "Wall" or name_collider == "Wall") and (name_collider[:6] == "Joueur" or name_collided[:6] == "Joueur"):
                    for player in self.players:
                        if player.name == name_collided or player.name == name_collider and player.onRace:
                            player.x = player.tx
                            player.model.setX(player.x)
                            player.y = player.ty
                            player.model.setY(player.y)
                            player.z = player.tz
                            player.model.setZ(player.z)

                if (name_collided[:-2] == "Score" or name_collider[:-2] == "Score") and (name_collider[:4] == "Joue" or name_collided[:4] == "Joue"):
                    for player in self.players:
                        if player.name == name_collided or player.name == name_collider and player.onRace:
                            if name_collided[:-2] == "Score":
                                if player.score > int(name_collided[-2:])-5:
                                    if player.score < int(name_collided[-2:]):
                                        player.score = int(name_collided[-2:])
                                        player.static_count = 0
                            else:
                                if player.score > int(name_collider[-2:])-5:
                                    if player.score < int(name_collider[-2:]):
                                        player.score = int(name_collider[-2:])
                                        player.static_count = 0

                if (name_collided[:-3] == "InclineP" or name_collider[:-3] == "InclineP") and (name_collider[:6] == "Joueur" or name_collided[:6] == "Joueur"):
                    for player in self.players:
                        if player.name == name_collided or player.name == name_collider and player.onRace:
                            if name_collided[:-3] == "InclineP":
                                player.model.setR(int(name_collided[-3:]))
                            else:
                                player.model.setR(int(name_collider[-3:]))

                if (name_collided[:-3] == "InclineR" or name_collider[:-3] == "InclineR") and (name_collider[:6] == "Joueur" or name_collided[:6] == "Joueur"):
                    for player in self.players:
                        if player.name == name_collided or player.name == name_collider and player.onRace:
                            if name_collided[:-3] == "InclineR":
                                player.model.setP(int(name_collided[-3:]))
                            else:
                                player.model.setP(int(name_collider[-3:]))

                if (name_collided[:-2] == "Boost" or name_collider[:-2] == "Boost") and (name_collider[:6] == "Joueur" or name_collided[:6] == "Joueur"):
                    for player in self.players:
                        if player.name == name_collided or player.name == name_collider and player.onRace:
                            if name_collided[:-2] == "Boost":
                                player.boost = int(name_collided[-2:])
                            else:
                                player.boost = int(name_collider[-2:])

                if (name_collided == "JumpPad" or name_collider == "JumpPad") and (name_collider[:6] == "Joueur" or name_collided[:6] == "Joueur"):
                    for player in self.players:
                        if player.name == name_collided or player.name == name_collider and player.onRace:
                            player.vz = 125

                if (name_collided == "Win" or name_collider == "Win") and (name_collider[:6] == "Joueur" or name_collided[:6] == "Joueur"):
                    for player in self.players:
                        if player.name == name_collided or player.name == name_collider and player.onRace:
                            if player.score == 68:
                                print(f"Victoire ! Temps : {None}")
        for player in self.players:
            if player.onRace:
                player.z += player.vz * dt
                player.model.setZ(player.model.getZ() + player.vz * dt)

    def initBackground(self):
        bgTexture = self.loader.loadTexture("maps/galaxy2.jpg")
        screenStage = TextureStage('screen')
        screenStage.setMode(TextureStage.MDecal)
        screenTexture = Texture()
        buffer = self.win.makeTextureBuffer("screen buffer", self.win.getXSize(), self.win.getYSize(),screenTexture, True)
        bufferCam = self.makeCamera(buffer, lens=self.cam.node().getLens())
        cm = CardMaker('screencard')
        cm.setFrameFullscreenQuad()
        cm.setHasUvs(True)
        screenCard = self.render2d.attachNewNode(cm.generate())
        screenCard.setTexture(bgTexture)
        screenCard.setTexture(screenStage, screenTexture)

    def initColliders(self):
        self.handler = CollisionHandlerQueue()
        self.traverserMap = CollisionTraverser()
        self.traverserWalls = CollisionTraverser()
        self.traverserBoosts = CollisionTraverser()
        self.traverserScores = CollisionTraverser()
        self.traverserInclinesP = CollisionTraverser()
        self.traverserInclinesR = CollisionTraverser()
        self.traverserJumpPads = CollisionTraverser()
        self.traverserWins = CollisionTraverser()
        self.traverserFly = CollisionTraverser()

        self.scoreNodePath = BitMask32.bit(2)
        self.mapNodePath = BitMask32.bit(1)
        self.wallsNodePath = BitMask32.bit(3)
        self.boostsNodePath = BitMask32.bit(4)
        self.inclinesPNodePath = BitMask32.bit(5)
        self.inclinesRNodePath = BitMask32.bit(6)
        self.jumpPadsNodePath = BitMask32.bit(7)
        self.winsNodePath = BitMask32.bit(8)

        self.walls = []
        self.boosts = []
        self.scores = []
        self.inclinesP = [] #Gauche
        self.inclinesR = [] #Avant
        self.jumpPads = []
        self.wins = []
        display_mode = False

        self.addCube(self.boosts,"Boost60",-384,-95.5,166,1,5,5,display_mode,self.traverserBoosts,self.boostsNodePath)
        self.addCube(self.inclinesR,"InclineR-10",-275,-95.5,137,1,7,5,display_mode,self.traverserInclinesR,self.inclinesRNodePath)
        self.addCube(self.inclinesR,"InclineR025",-220,-95.5,124,1,7,5,display_mode,self.traverserInclinesR,self.inclinesRNodePath)
        self.addCube(self.inclinesP,"InclineP025",-156,-101.5,147,1,7,5,display_mode,self.traverserInclinesP,self.inclinesPNodePath)

        self.addCube(self.inclinesR,"InclineR000",-214,-96.5,168,7,1,5,display_mode,self.traverserInclinesR,self.inclinesRNodePath)
        self.addCube(self.boosts,"Boost60",-214,-96.5,168,7,1,5,display_mode,self.traverserBoosts,self.boostsNodePath)
        self.addCube(self.inclinesP,"InclineP000",-214,-96.5,168,7,1,5,display_mode,self.traverserInclinesP,self.inclinesPNodePath)
        self.addCube(self.boosts,"Boost60",-59,-1,158,10,6,5,display_mode,self.traverserBoosts,self.boostsNodePath)
        self.addCube(self.boosts,"InclineR-05",-59,-1,158,10,6,5,display_mode,self.traverserBoosts,self.inclinesRNodePath)
        self.addCube(self.boosts,"Boost60",123,95,81,1,7,5,display_mode,self.traverserBoosts,self.boostsNodePath)
        self.addCube(self.inclinesR,"InclineR000",123,95,81,1,7,5,display_mode,self.traverserInclinesR,self.inclinesRNodePath)
        self.addCube(self.boosts,"Boost60",168,76,72,20,10,5,display_mode,self.traverserBoosts,self.boostsNodePath)
        self.addCube(self.boosts,"Boost60",230,56,75,2,2,3,display_mode,self.traverserBoosts,self.boostsNodePath)
        self.addCube(self.boosts,"Boost30",266,59,88,2,2,3,display_mode,self.traverserBoosts,self.boostsNodePath)
        self.addCube(self.boosts,"Boost30",238,104,98,2,2,3,display_mode,self.traverserBoosts,self.boostsNodePath)
        self.addCube(self.boosts,"Boost30",211,70,102,2,2,3,display_mode,self.traverserBoosts,self.boostsNodePath)
        self.addCube(self.jumpPads,"JumpPad",196,43,118,7,3,8,display_mode,self.traverserJumpPads,self.jumpPadsNodePath)
        self.addCube(self.boosts,"Boost70",196,43,118,7,3,8,display_mode,self.traverserBoosts,self.boostsNodePath)

        self.addCube(self.inclinesR,"InclineR000",196,43,118,7,3,8,display_mode,self.traverserInclinesR,self.inclinesRNodePath)
        self.addCube(self.inclinesP,"InclineP000",196,43,118,7,3,8,display_mode,self.traverserInclinesP,self.inclinesPNodePath)
        self.addCube(self.inclinesP,"InclineP-30",200,63,70,1,7,5,display_mode,self.traverserInclinesP,self.inclinesPNodePath)
        self.addCube(self.inclinesR,"InclineR020",200,63,70,1,7,5,display_mode,self.traverserInclinesR,self.inclinesRNodePath)

        self.addCube(self.inclinesR,"InclineR-15",91,-224,276,7,1,5,display_mode,self.traverserInclinesR,self.inclinesRNodePath)
        self.addCube(self.inclinesR,"InclineR015",65,-295,233,7,1,5,display_mode,self.traverserInclinesR,self.inclinesRNodePath)
        self.addCube(self.boosts,"Boost60",60,-309,236,7,1,3,display_mode,self.traverserBoosts,self.boostsNodePath)
        self.addCube(self.inclinesP,"InclineP-30",46,-352,239,7,1,5,display_mode,self.traverserInclinesP,self.inclinesPNodePath)
        self.addCube(self.inclinesR,"InclineR000",43,-355,240,7,1,5,display_mode,self.traverserInclinesR,self.inclinesRNodePath)
        self.addCube(self.inclinesR,"InclineR-35",48,-397,238,2,9,7,display_mode,self.traverserInclinesR,self.inclinesRNodePath)
        self.addCube(self.boosts,"Boost10",84,-407,212,1,7,1,display_mode,self.traverserBoosts,self.boostsNodePath)
        self.addCube(self.boosts,"Boost10",91,-407,207,1,7,1,display_mode,self.traverserBoosts,self.boostsNodePath)
        self.addCube(self.boosts,"Boost10",101,-408,202,1,7,1,display_mode,self.traverserBoosts,self.boostsNodePath)
        self.addCube(self.inclinesP,"InclineP000",134,-363,174,1,9,5,display_mode,self.traverserInclinesP,self.inclinesPNodePath)
        self.addCube(self.inclinesR,"InclineR000",104,-346,158,1,9,5,display_mode,self.traverserInclinesR,self.inclinesRNodePath)
        self.addCube(self.boosts,"Boost60",91,-343,157,1,9,3,display_mode,self.traverserBoosts,self.boostsNodePath)
        self.addCube(self.inclinesP,"InclineP010",-70,-349,147,1,11,5,display_mode,self.traverserInclinesP,self.inclinesPNodePath)
        self.addCube(self.inclinesP,"InclineP000",-114,-272,141,1,9,5,display_mode,self.traverserInclinesP,self.inclinesPNodePath)
        self.addCube(self.inclinesP,"InclineR010",-114,-272,141,1,9,5,display_mode,self.traverserInclinesR,self.inclinesRNodePath)

        self.addCube(self.inclinesP,"InclineP005",-153,-350,161,8,3,5,display_mode,self.traverserInclinesP,self.inclinesPNodePath)
        self.addCube(self.inclinesP,"InclineP000",-219,-324,166,3,6,5,display_mode,self.traverserInclinesP,self.inclinesPNodePath)
        self.addCube(self.boosts,"Boost60",-219,-324,166,3,6,5,display_mode,self.traverserBoosts,self.boostsNodePath)


        self.addCube(self.boosts,"Boost60",-282,-294,160,2,7,3,display_mode,self.traverserBoosts,self.boostsNodePath)
        self.addCube(self.boosts,"Boost60",-272,-279,160,2,7,3,display_mode,self.traverserBoosts,self.boostsNodePath)

        self.addCube(self.boosts,"Boost60",-353,-262,156,2,7,3,display_mode,self.traverserBoosts,self.boostsNodePath)
        self.addCube(self.boosts,"Boost60",-342,-248,156,2,7,3,display_mode,self.traverserBoosts,self.boostsNodePath)
        self.addCube(self.boosts,"Boost60",-334,-233,156,2,7,3,display_mode,self.traverserBoosts,self.boostsNodePath)

        self.addCube(self.boosts,"Boost60",-410,-215,153,2,7,3,display_mode,self.traverserBoosts,self.boostsNodePath)
        self.addCube(self.boosts,"Boost60",-401,-201,153,2,7,3,display_mode,self.traverserBoosts,self.boostsNodePath)

        self.addCube(self.inclinesR, "InclineR-08", -410, -215, 153, 2, 7, 3, display_mode, self.traverserInclinesR,self.inclinesRNodePath)
        self.addCube(self.inclinesR, "InclineR-08", -401, -201, 153, 2, 7, 3, display_mode, self.traverserInclinesR,self.inclinesRNodePath)

        self.addCube(self.inclinesP,"InclineR030",-469,-170,142,8,13,8,display_mode,self.traverserInclinesR,self.inclinesRNodePath)
        self.addCube(self.inclinesP,"InclineP030",-576,-171,253,1,9,5,display_mode,self.traverserInclinesP,self.inclinesPNodePath)
        self.addCube(self.inclinesR, "InclineR-30", -588, -146, 269, 8, 8, 7, display_mode, self.traverserInclinesR,self.inclinesRNodePath)
        self.addCube(self.inclinesP,"InclineP000",-567,-118,245,1,7,15,display_mode,self.traverserInclinesP,self.inclinesPNodePath)
        self.addCube(self.inclinesP,"InclineR-05",-499,-138,187,1,9,5,display_mode,self.traverserInclinesR,self.inclinesRNodePath)
        self.addCube(self.inclinesP,"InclineP-05",-487,-75,174,1,16,8,display_mode,self.traverserInclinesP,self.inclinesPNodePath)
        self.addCube(self.boosts,"Boost60",-476,-151,171,1,3,3,display_mode,self.traverserBoosts,self.boostsNodePath)
        self.addCube(self.boosts,"Boost60",-444,-109,170,2,1,3,display_mode,self.traverserBoosts,self.boostsNodePath)
        self.addCube(self.inclinesP,"InclineP000",-451,-81,168,7,1,5,display_mode,self.traverserInclinesP,self.inclinesPNodePath)
        self.addCube(self.inclinesP,"InclineR000",-451,-81,168,7,1,5,display_mode,self.traverserInclinesR,self.inclinesRNodePath)
        self.addCube(self.wins,"Win",-451,-61,168,7,1,5,display_mode,self.traverserWins,self.winsNodePath)

        self.addSegment(self.walls,"Wall",-252,-28,155,-225,-89,155,display_mode,self.traverserWalls,self.wallsNodePath)
        self.addSegment(self.walls,"Wall",-236,-22,155,-209, -82, 155,display_mode,self.traverserWalls,self.wallsNodePath)

        self.addSegment(self.walls,"Wall",132,-87,278,97, -185, 281,display_mode,self.traverserWalls,self.wallsNodePath)
        self.addSegment(self.walls,"Wall",148,-93,279,112, -192, 281,display_mode,self.traverserWalls,self.wallsNodePath)

        self.addSegment(self.walls,"Wall",-497,-146,186,-483, -141, 181,display_mode,self.traverserWalls,self.wallsNodePath)
        self.addSegment(self.walls,"Wall",-483,-141,181,-469, -128, 178,display_mode,self.traverserWalls,self.wallsNodePath)
        self.addSegment(self.walls,"Wall",-469,-128,178,-465, -116, 177,display_mode,self.traverserWalls,self.wallsNodePath)
        self.addSegment(self.walls,"Wall",-465,-116,177,-462, -103, 175,display_mode,self.traverserWalls,self.wallsNodePath)
        self.addSegment(self.walls,"Wall",-462,-103,175,-462, -89, 174,display_mode,self.traverserWalls,self.wallsNodePath)
        self.addSegment(self.walls,"Wall",-462,-89,174,-466, -76, 174,display_mode,self.traverserWalls,self.wallsNodePath)
        self.addSegment(self.walls,"Wall",-466,-76,174,-466, -66, 177,display_mode,self.traverserWalls,self.wallsNodePath)
        self.addSegment(self.walls,"Wall",-466,-66,177,-474, -59, 178,display_mode,self.traverserWalls,self.wallsNodePath)
        self.addSegment(self.walls,"Wall",-474,-59,178,-496, -59, 179,display_mode,self.traverserWalls,self.wallsNodePath)

        self.addCube(self.scores,"Score01",-450,-37,168,7,7,5,display_mode,self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores,"Score02",-451,5,168,7,7,5,display_mode,self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores,"Score03",-413,20,168,7,7,5,display_mode,self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores,"Score04",-376,20,168,7,7,5,display_mode,self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores,"Score05",-365,-19,168,7,7,5,display_mode,self.traverserScores,self.scoreNodePath)

        self.addCube(self.scores, "Score06", -318, -42, 168, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score07", -363, -51, 168, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score08", -398, -62, 168, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score09", -374, -95, 168, 2, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score10", -283, -95, 141, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)

        self.addCube(self.scores, "Score11", -230, -95, 121, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score12", -171, 98, 138, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score13", -155, -101, 145, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score14", -135, -163, 172, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score15", -177, -171, 165, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)

        self.addCube(self.scores, "Score16", -199, -136, 161, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score17", -211, -100, 166, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score18", -237, -36, 156, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score19", -233, -2, 156, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score20", -201, 24, 156, 8, 8, 5, display_mode, self.traverserScores,self.scoreNodePath)

        self.addCube(self.scores, "Score21", -221, 91, 156, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score22", -197, 154, 156, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score23", -160, 99, 156, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score24", -125, 22, 156, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score25", -89, -45, 156, 9, 9, 5, display_mode, self.traverserScores,self.scoreNodePath)

        self.addCube(self.scores, "Score26", -61, -3, 159, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score27", -9, 68, 132, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score28", 58, 70, 114, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score29", 111, 66, 102, 9, 9, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score30", 64, 90, 93, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)

        self.addCube(self.scores, "Score31", 44, 115, 96, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        # self.addCube(self.scores, "Score32", 122, 95, 77, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score33", 180, 72, 68, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score34", 225, 54, 74, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score35", 261, 59, 86, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)

        self.addCube(self.scores, "Score36", 249, 100, 94, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score37", 216, 83, 103, 7, 7, 6, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score38", 197, 45, 117, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score39", 127, -124, 279, 12, 12, 20, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score40", 99, -208, 279, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)

        self.addCube(self.scores, "Score41", 86, -244, 261, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score42", 67, -290, 232, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score43", 49, -340, 239, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score44", 46, -393, 238, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score45", 89, -408, 210, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)

        self.addCube(self.scores, "Score46", 137, -404, 190, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score47", 138, -366, 177, 7, 7, 8, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score48", 93, -345, 156, 8, 8, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score49", 31, -311, 146, 12, 12, 12, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score50", -12, -330, 140, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)

        self.addCube(self.scores, "Score51", -65, -351, 146, 9, 9, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score52", -93, -299, 142, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score53", -135, -278, 145, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score54", -148, -340, 157, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score55", -217, -325, 167, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)

        self.addCube(self.scores, "Score56", -281, -296, 161, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score56", -271, -280, 161, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score57", -352, -263, 157, 6, 6, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score57", -342, -247, 157, 6, 6, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score57", -334, -233, 157, 6, 6, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score58", -409, -213, 152, 6, 6, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score58", -400, -200, 152, 6, 6, 5, display_mode, self.traverserScores,self.scoreNodePath)

        self.addCube(self.scores, "Score59", -456, -176, 138, 12, 12, 12, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score60", -527, --167, 193, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)

        self.addCube(self.scores, "Score61", -536, -168, 203, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score62", -588, -146, 269, 8, 8, 7, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score63", -555, -119, 233, 7, 7, 8, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score64", -496, -136, 195, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score65", -472, -96, 182, 7, 7, 5, display_mode, self.traverserScores,self.scoreNodePath)

        self.addCube(self.scores, "Score66", -511, -79, 181, 10, 10, 9, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score67", -484, -147, 168, 13, 13, 5, display_mode, self.traverserScores,self.scoreNodePath)
        self.addCube(self.scores, "Score68", -449, -103, 168, 10, 10, 5, display_mode, self.traverserScores,self.scoreNodePath)

        self.addSegment(self.walls, "Wall", -40, -359, 140, -50, -364, 142, display_mode, self.traverserWalls,self.wallsNodePath)
        self.addSegment(self.walls, "Wall", -50, -364, 142, -67, -366, 146, display_mode, self.traverserWalls,self.wallsNodePath)
        self.addSegment(self.walls, "Wall", -67, -366, 146, -80, -360, 150, display_mode, self.traverserWalls,self.wallsNodePath)
        self.addSegment(self.walls, "Wall", -80, -360, 150, -91, -348, 151, display_mode, self.traverserWalls,self.wallsNodePath)
        self.addSegment(self.walls, "Wall", -91, -348, 151, -94, -336, 149, display_mode, self.traverserWalls,self.wallsNodePath)
        self.addSegment(self.walls, "Wall", -94, -336, 149, -97, -318, 146, display_mode, self.traverserWalls,self.wallsNodePath)

    def addCube(self,tab,tag,x,y,z,dx,dy,dz,display,traverser,nodePath):
        cube = CollisionBox((x,y,z),dx,dy,dz)
        node = CollisionNode(tag)
        node.addSolid(cube)
        node.setFromCollideMask(0)
        node.setIntoCollideMask(nodePath)
        path = self.render.attachNewNode(node)
        path.setPythonTag(tag, node)
        traverser.addCollider(path, self.handler)
        if display:
            path.show()
        tab.append(node)

    def addSegment(self,tab,tag,xa,ya,za,xb,yb,zb,display,traverser,nodePath):
        segment = CollisionSegment(xa,ya,za,xb,yb,zb)
        node = CollisionNode(tag)
        node.addSolid(segment)
        node.setFromCollideMask(0)
        node.setIntoCollideMask(nodePath)
        path = self.render.attachNewNode(node)
        path.setPythonTag(tag, node)
        traverser.addCollider(path, self.handler)
        if display:
            path.show()
        tab.append(node)

    def addPolygon(self,tab,tag,x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4,display,traverser,nodePath):
        polygon = CollisionPolygon(LPoint3f(x1,y1,z1),
                                   LPoint3f(x2, y2, z2),
                                   LPoint3f(x3, y3, z3),
                                   LPoint3f(x4, y4, z4)
                                   )
        node = CollisionNode(tag)
        node.addSolid(polygon)
        node.setFromCollideMask(0)
        node.setIntoCollideMask(nodePath)
        path = self.render.attachNewNode(node)
        path.setPythonTag(tag,node)
        traverser.addCollider(path,self.handler)
        if display:
            path.show()
        tab.append(node)

    def createPlayerNode(self,x,y,z,name):
        playerNode = self.render.attachNewNode("Player")
        playerNode.setPos(x, y, z)
        playerNode.setScale(3)

        if x == self.spawnPoint1[0]:
            playerNode.setH(0)
        if x == self.spawnPoint2[0]:
            playerNode.setH(-20)
        if x == self.spawnPoint3[0]:
            playerNode.setH(-105)
        if x == self.spawnPoint4[0]:
            playerNode.setH(-195)
        if x == self.spawnPoint5[0]:
            playerNode.setH(55)


        self.playerModel.instanceTo(playerNode)

        collisionBox = CollisionBox((-0.55, -0.55, -0), (0.55, 0.9, 1))
        collisionNode = CollisionNode(f"Joueur{name}")
        collisionNode.addSolid(collisionBox)
        collider = playerNode.attachNewNode(collisionNode)
        collider.setPythonTag("Joueur", playerNode)

        collisionFlyBox = CollisionBox((-0.55, -0.55, -0.1), (0.55, 0.9, 1))
        collisionFlyNode = CollisionNode(f"JoueurFly{name}")
        collisionFlyNode.addSolid(collisionFlyBox)
        colliderFly = playerNode.attachNewNode(collisionFlyNode)
        colliderFly.setPythonTag("JoueurFly", playerNode)

        cBox = CollisionBox((-2.3, 4, -3.5), (-2.5, 4.2, 3))
        collisionDLNode = CollisionNode(f"JoueurDangerLeft{name}")
        collisionDLNode.addSolid(cBox)
        colliderDL = playerNode.attachNewNode(collisionDLNode)
        colliderDL.setPythonTag("Joueur", playerNode)

        cBox = CollisionBox((2.3, 4, -3.5), (2.5, 4.2, 3))
        collisionDRNode = CollisionNode(f"JoueurDangerRight{name}")
        collisionDRNode.addSolid(cBox)
        colliderDR = playerNode.attachNewNode(collisionDRNode)
        colliderDR.setPythonTag("Joueur", playerNode)

        cBox = CollisionBox((-0.7, -0.2, -3.5), (-0.9, 0.2, 3))
        collisionDL2Node = CollisionNode(f"JoueurDangerLeft2{name}")
        collisionDL2Node.addSolid(cBox)
        colliderDL2 = playerNode.attachNewNode(collisionDL2Node)
        colliderDL2.setPythonTag("Joueur", playerNode)

        cBox = CollisionBox((0.7, -0.2, -3.5), (0.9, 0.2, 3))
        collisionDR2Node = CollisionNode(f"JoueurDangerRight2{name}")
        collisionDR2Node.addSolid(cBox)
        colliderDR2 = playerNode.attachNewNode(collisionDR2Node)
        colliderDR2.setPythonTag("Joueur", playerNode)

        cBox = CollisionBox((-7, 4, -3.5), (-7.2, 4.2, 3))
        collisionRLNode = CollisionNode(f"JoueurRoadLeft{name}")
        collisionRLNode.addSolid(cBox)
        colliderRL = playerNode.attachNewNode(collisionRLNode)
        colliderRL.setPythonTag("Joueur", playerNode)

        cBox = CollisionBox((7, 4, -3.5), (7.2, 4.2, 3))
        collisionRRNode = CollisionNode(f"JoueurRoadRight{name}")
        collisionRRNode.addSolid(cBox)
        colliderRR = playerNode.attachNewNode(collisionRRNode)
        colliderRR.setPythonTag("Joueur", playerNode)

        cBox = CollisionBox((-2, 30, -40), (2, 120, 220))
        collisionFrontNode = CollisionNode(f"JoueurFront{name}")
        collisionFrontNode.addSolid(cBox)
        colliderFront = playerNode.attachNewNode(collisionFrontNode)
        colliderFront.setPythonTag("Joueur", playerNode)

        cBox = CollisionBox((-0.2, 1, -3.5), (0.2, 1.3, 3))
        collisionFDNode = CollisionNode(f"JoueurFrontDanger{name}")
        collisionFDNode.addSolid(cBox)
        colliderFD = playerNode.attachNewNode(collisionFDNode)
        colliderFD.setPythonTag("Joueur", playerNode)

        cBox = CollisionBox((-0.3, 1.7, -3.5), (0.3, 2, 3))
        collisionFD2Node = CollisionNode(f"JoueurFrontDanger2{name}")
        collisionFD2Node.addSolid(cBox)
        colliderFD2 = playerNode.attachNewNode(collisionFD2Node)
        colliderFD2.setPythonTag("Joueur", playerNode)

        cBox = CollisionBox((-4.7, 4, -3.5), (-5, 4.2, 3))
        collisionRLSNode = CollisionNode(f"JoueurRoadLeftSpe{name}")
        collisionRLSNode.addSolid(cBox)
        colliderRLS = playerNode.attachNewNode(collisionRLSNode)
        colliderRLS.setPythonTag("Joueur", playerNode)

        cBox = CollisionBox((4.7, 4, -3.5), (5, 4.2, 3))
        collisionRRSNode = CollisionNode(f"JoueurRoadRightSpe{name}")
        collisionRRSNode.addSolid(cBox)
        colliderRRS = playerNode.attachNewNode(collisionRRSNode)
        colliderRRS.setPythonTag("Joueur", playerNode)

        self.traverserWins.addCollider(collider, self.handler)
        self.traverserBoosts.addCollider(collider, self.handler)
        self.traverserJumpPads.addCollider(collider, self.handler)
        self.traverserWalls.addCollider(collider, self.handler)
        self.traverserInclinesP.addCollider(collider, self.handler)
        self.traverserInclinesR.addCollider(collider, self.handler)
        self.traverserScores.addCollider(collider, self.handler)
        self.traverserMap.addCollider(collider, self.handler)
        self.traverserFly.addCollider(colliderFly, self.handler)

        self.traverserMap.addCollider(colliderDL,self.handler)
        self.traverserMap.addCollider(colliderRL,self.handler)
        self.traverserMap.addCollider(colliderDR,self.handler)
        self.traverserMap.addCollider(colliderRR,self.handler)
        self.traverserMap.addCollider(colliderFront,self.handler)
        self.traverserMap.addCollider(colliderFD,self.handler)
        self.traverserMap.addCollider(colliderFD2,self.handler)
        self.traverserMap.addCollider(colliderDL2,self.handler)
        self.traverserMap.addCollider(colliderDR2,self.handler)
        self.traverserMap.addCollider(colliderRLS, self.handler)
        self.traverserMap.addCollider(colliderRRS, self.handler)



        collisionDLNode.setFromCollideMask(BitMask32.bit(1))
        collisionDRNode.setFromCollideMask(BitMask32.bit(1))
        collisionRLNode.setFromCollideMask(BitMask32.bit(1))
        collisionRRNode.setFromCollideMask(BitMask32.bit(1))
        collisionFrontNode.setFromCollideMask(BitMask32.bit(1))
        collisionFDNode.setFromCollideMask(BitMask32.bit(1))
        collisionFD2Node.setFromCollideMask(BitMask32.bit(1))
        collisionDL2Node.setFromCollideMask(BitMask32.bit(1))
        collisionDR2Node.setFromCollideMask(BitMask32.bit(1))
        collisionRLSNode.setFromCollideMask(BitMask32.bit(1))
        collisionRRSNode.setFromCollideMask(BitMask32.bit(1))

        collisionDLNode.setIntoCollideMask(BitMask32.bit(9))
        collisionDRNode.setIntoCollideMask(BitMask32.bit(9))
        collisionRLNode.setIntoCollideMask(BitMask32.bit(9))
        collisionRRNode.setIntoCollideMask(BitMask32.bit(9))
        collisionFrontNode.setIntoCollideMask(BitMask32.bit(9))
        collisionFDNode.setIntoCollideMask(BitMask32.bit(9))
        collisionFD2Node.setIntoCollideMask(BitMask32.bit(9))
        collisionDL2Node.setIntoCollideMask(BitMask32.bit(9))
        collisionDR2Node.setIntoCollideMask(BitMask32.bit(9))
        collisionRLSNode.setIntoCollideMask(BitMask32.bit(9))
        collisionRRSNode.setIntoCollideMask(BitMask32.bit(9))

        # Afficher les Perceptrons
        # colliderRL.show()
        # colliderDL.show()
        # colliderDR.show()
        # colliderRR.show()
        # colliderFD.show()
        # colliderDR2.show()
        # colliderDL2.show()
        # colliderFront.show()
        # colliderRRS.show()
        # colliderRLS.show()

        # colliderFD2.show()

        collisionFlyNode.setFromCollideMask(BitMask32.bit(1))
        collisionNode.setFromCollideMask(BitMask32.bit(1) | BitMask32.bit(2) | BitMask32.bit(3) | BitMask32.bit(4) | BitMask32.bit(5) | BitMask32.bit(6) | BitMask32.bit(7) | BitMask32.bit(8))

        collisionFlyNode.setIntoCollideMask(BitMask32.bit(9))
        collisionNode.setIntoCollideMask(BitMask32.bit(9))

        cshow = [colliderRL,colliderFD,colliderRR,colliderDL2,colliderFD2,colliderRRS,colliderRLS,colliderDL,colliderDR2,colliderFront]

        return playerNode,cshow

    def initMap(self):
        self.mapNode = self.render.attachNewNode("Map")
        self.mapNode.setPos(0,0,0)
        self.mapNode.setScale(6)
        self.mapNode.setHpr(90,90,90)
        self.map.instanceTo(self.mapNode)
        self.cshapes = False

        model_copy = self.map.copy_to(self.render)
        model_copy.detach_node()
        model_copy.flatten_light()
        collision_root = NodePath("Collision Root")
        collision_root.reparent_to(self.map)

        for model in model_copy.find_all_matches("**/+GeomNode"):
            model_node = model.node()
            collision_node = CollisionNode(model_node.name)

            collision_node.setIntoCollideMask(BitMask32.bit(1))
            collision_node.setFromCollideMask(BitMask32.bit(0))

            collision_mesh = collision_root.attach_new_node(collision_node)
            collision_mesh.setPythonTag("Map", model_node)
            self.traverserMap.addCollider(collision_mesh, self.handler)
            self.traverserFly.addCollider(collision_mesh, self.handler)
            # collision_mesh.show()

            for geom in model_node.modify_geoms():
                geom.decompose_in_place()
                vertex_data = geom.modify_vertex_data()
                vertex_data.format = GeomVertexFormat.get_v3()
                view = memoryview(vertex_data.arrays[0]).cast("B").cast("f")
                index_list = geom.primitives[0].get_vertex_list()
                index_count = len(index_list)

                for indices in (index_list[i:i + 3] for i in range(0, index_count, 3)):
                    points = [Point3(*view[index * 3:index * 3 + 3]) for index in indices]
                    coll_poly = CollisionPolygon(*points)
                    collision_node.add_solid(coll_poly)

    def updateCamera(self,task):
        dt = globalClock.getDt()

        v = 28 #Vitesse de translation
        omega = 25 #Vitesse de Rotation
        dx = 0
        dy = 0
        dz = 0

        if self.keyMap['forward']:
            dx -= dt * v * sin(degToRad(self.camera.getH()))
            dy += dt * v * cos(degToRad(self.camera.getH()))
        if self.keyMap['backward']:
            dx += dt * v * sin(degToRad(self.camera.getH()))
            dy -= dt * v * cos(degToRad(self.camera.getH()))
        if self.keyMap['left']:
            dx -= dt * v * cos(degToRad(self.camera.getH()))
            dy -= dt * v * sin(degToRad(self.camera.getH()))
        if self.keyMap['right']:
            dx += dt * v * cos(degToRad(self.camera.getH()))
            dy += dt * v * sin(degToRad(self.camera.getH()))
        if self.keyMap['up']:
            dz += dt * v
        if self.keyMap['down']:
            dz -= dt * v

        camera = self.camera
        camera.setPos(
            camera.getX() + dx,
            camera.getY() + dy,
            camera.getZ() + dz)

        if self.into_screen:
            coordSouris = self.win.getPointer(0)
            mouseX = coordSouris.getX()
            mouseY = coordSouris.getY()

            mouseChangeX = mouseX - self.lastMouseX
            mouseChangeY = mouseY - self.lastMouseY

            currentH = self.camera.getH()
            currentP = self.camera.getP()

            self.camera.setHpr(
                currentH - mouseChangeX * dt * omega,
                min(90, max(-90, currentP - mouseChangeY * dt * omega)),
                0
            )

            self.lastMouseX = mouseX
            self.lastMouseY = mouseY

        if self.cameraStick and self.players[self.idx].onRace:
            self.camera.setPos(
                self.players[self.idx].model.getX() - 7*cos(degToRad(self.players[self.idx].model.getH() + 90)),
                self.players[self.idx].model.getY() - 7*sin(degToRad(self.players[self.idx].model.getH() + 90)),
                self.players[self.idx].model.getZ() + 5,
            )
        # print(self.camera.getPos())
        return task.cont

    def importModels(self):
        self.map = self.loader.loadModel("maps/rainbow_road_dxMODEL.glb")
        self.deco = self.loader.loadModel("maps/rainbow_road_dxDECO.glb")
        self.deco.setPos(0, 0, 0)
        self.deco.setScale(6)
        self.deco.setHpr(90, 90, 90)
        self.deco.reparentTo(self.render)
        self.playerModel = self.loader.loadModel("player/mario_kart.glb")
        self.playerModel.setMaterialOff(1)
        self.playerModel.setHpr(90,90,90)

    def initCamera(self):
        self.into_screen = True
        self.cameraStick = False
        self.onscreen = True

        self.disableMouse()
        self.camLens.setFov(110)
        self.camera.setPos(-451.135, -71.7493, 173.361)
        self.captureMouse()

        self.lastMouseX = 0
        self.lastMouseY = 0


    def initLights(self):
        render = self.render

        mainLight = DirectionalLight("MainLight")
        mainLightNodePath = render.attachNewNode(mainLight)
        mainLightNodePath.setHpr(30, -60, 0)
        render.setLight(mainLightNodePath)

        ambientLight = AmbientLight("AmbientLight")
        ambientLight.setColor((0.6, 0.6, 0.6, 1))
        ambientLightNodePath = render.attachNewNode(ambientLight)
        render.setLight(ambientLightNodePath)

    def initControls(self):
        self.keyMap = {
            "forward": False,
            "left": False,
            "right": False,
            "backward": False,
            "down": False,
            "up": False
        }

        self.accept("z", self.updateKeyMap, ["forward", True])
        self.accept("q", self.updateKeyMap, ["left", True])
        self.accept("s", self.updateKeyMap, ["backward", True])
        self.accept("d", self.updateKeyMap, ["right", True])
        self.accept("lshift", self.updateKeyMap, ["down", True])
        self.accept("space", self.updateKeyMap, ["up", True])

        self.accept("z-up", self.updateKeyMap, ["forward", False])
        self.accept("q-up", self.updateKeyMap, ["left", False])
        self.accept("s-up", self.updateKeyMap, ["backward", False])
        self.accept("d-up", self.updateKeyMap, ["right", False])
        self.accept("lshift-up", self.updateKeyMap, ["down", False])
        self.accept("space-up", self.updateKeyMap, ["up", False])

        self.accept("a",self.stickCamera)
        self.accept("e",self.displayTicks)
        self.accept("r",self.reduceTicks)
        self.accept("t",self.increaseTicks)

        self.accept("mouse1", self.captureMouse)
        self.accept("mouse3", self.releaseMouse)

        self.accept("mouse4", self.offScreen)
        self.accept("mouse5", self.onScreen)

        self.accept("p",self.plotCurrentStats)
        self.accept("n",self.increaseIdx)
        self.accept("b",self.decreaseIdx)
        self.accept("v",self.displayCShapes)

    def displayCShapes(self):
        self.cshapes = not self.cshapes
        if self.cshapes:
            for player in self.players:
                for c in player.cshow:
                    c.show()
        else:
            for player in self.players:
                for c in player.cshow:
                    c.hide()

    def increaseIdx(self):
        self.idx = min(self.idx + 1,19)
        print(f"Joueur Spec : {self.idx}")

    def decreaseIdx(self):
        self.idx = max(0,self.idx-1)
        print(f"Joueur Spec : {self.idx}")

    def plotCurrentStats(self):
        plt.figure()
        plt.plot(self.best_fitnesses, label="Meilleure Fitness")
        plt.plot(self.mean_fitnesses,label = "Fitness Moyenne")
        plt.title("Progression des IA")
        plt.xlabel("Parties")
        plt.ylabel("Qualité")
        plt.legend()
        plt.show()
    def displayTicks(self):
        print(f"Ticks actuels : {self.tick}, Ticks Max : {self.tickmax}")

    def reduceTicks(self):
        self.tickmax -= 10
        print(f"Ticks Max : {self.tickmax}")

    def increaseTicks(self):
        self.tickmax += 10
        print(f"Ticks Max : {self.tickmax}")

    def stickCamera(self):
        self.cameraStick = not self.cameraStick

    def offScreen(self):
        self.win.set_active(0)
        self.onscreen = False
        for player in self.players:
            player.static_count = 0
        print("Affichage du jeu OFF")

    def onScreen(self):
        self.win.set_active(1)
        self.onscreen = True
        for player in self.players:
            player.static_count = 0
        print("Affichage du jeu ON")

    def boostPlayer(self, player):
        if player.nbchampi > 0:
            player.boost = 60
            player.nbchampi -= 1

    def decide(self,player):
        perception = [player.frontDanger,player.dl2,player.dr2,player.dangerLeft,player.dangerRight,player.roadLeft,player.roadRight,player.front,player.rls,player.rrs]
        actions = player.ia.activate(perception)
        actions_mask = [actions[i] > 0 for i in range(3)]
        return actions_mask

    def moovePlayers(self,dt):

        self.tick += dt

        for i,player in enumerate(self.players):

            if player.onRace:
                if player.z < 50 or player.static_count > 9:
                    player.onRace = False
                    player.model.removeNode()
                    if self.cameraStick and self.idx == i:
                        for j in range(len(self.players)):
                            if self.players[j].onRace:
                                self.idx = j
                                break

            if player.onRace:
                actions_mask = self.decide(player)
                dx,dy,dz = 0,0,0

                player.tx = player.x
                player.ty = player.y
                player.tz = player.z

                if actions_mask[0]: #Avancer
                    dy += dt * player.v * sin(degToRad(player.model.getH() + 90))
                    dx += dt * player.v * cos(degToRad(player.model.getH() + 90))

                # if actions_mask[1]:
                #     dy -= dt * player.v * sin(degToRad(player.model.getH() + 90))
                #     dx -= dt * player.v * cos(degToRad(player.model.getH() + 90))

                if actions_mask[1]:
                    player.angle += player.w*dt
                    player.model.setH(player.model.getH() + player.w*dt)
                    if self.cameraStick and i == self.idx:
                        self.camera.setH(player.model.getH())

                if actions_mask[2]:
                    player.angle -= player.w*dt
                    player.model.setH(player.model.getH() - player.w*dt)
                    if self.cameraStick and i == self.idx:
                        self.camera.setH(player.model.getH())

                dy += dt * player.boost * sin(degToRad(player.model.getH() + 90))
                dx += dt * player.boost * cos(degToRad(player.model.getH() + 90))
                player.boost = max(0,player.boost - 20*dt)

                player.model.setPos(
                    player.model.getX() + dx*0.5,
                    player.model.getY() + dy*0.5,
                    player.model.getZ() + dz*0.5,
                )

                player.x += dx*0.5
                player.y += dy*0.5
                player.z += dz*0.5

    def updatePlayerKeyMap(self, key, value):
        self.playerKeyMap[key] = value
    def updateKeyMap(self,key,value):
        self.keyMap[key] = value

    def captureMouse(self):
        self.into_screen = True
        properties = WindowProperties()
        properties.setCursorHidden(True)
        properties.setMouseMode(WindowProperties.M_relative)
        self.win.requestProperties(properties)
        coordSouris = self.win.getPointer(0)
        self.lastMouseX = coordSouris.getX()
        self.lastMouseY = coordSouris.getY()

    def releaseMouse(self):
        self.into_screen = False
        properties = WindowProperties()
        properties.setCursorHidden(False)
        properties.setMouseMode(WindowProperties.M_relative)
        self.win.requestProperties(properties)

jeu = Game()
jeu.run()
