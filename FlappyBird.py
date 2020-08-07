import os
import random
import time

import neat
import pygame

WIN_WIDTH = 500
WIN_HEIGHT = 800

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))), pygame.transform.scale2x(
    pygame.image.load(os.path.join("imgs", "bird2.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))]
PIPE_IMG = pygame.transform.scale2x(
    pygame.image.load(os.path.join("imgs", "pipe.png")))
BASE_IMG = pygame.transform.scale2x(
    pygame.image.load(os.path.join("imgs", "base.png")))
BACK_IMG = pygame.transform.scale2x(
    pygame.image.load(os.path.join("imgs", "bg.png")))

pygame.font.init()
STAT_FONT = pygame.font.SysFont("comicsans", 50)


class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROTATION_VELOCITY = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.velocity = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.velocity = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1

        distance = self.velocity*self.tick_count + 1.5*self.tick_count**2

        if distance >= 16:
            distance = 16
        if distance < 0:
            distance -= 2

        self.y = self.y + distance

        # Setting tilt to fixed value upon reaching max height
        if distance < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        elif self.tilt > -90:
            self.tilt -= self.ROTATION_VELOCITY

    def draw(self, win):
        self.img_count += 1

        # Bird tilting up
        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]

        # Bird tilting down
        elif self.img_count < self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*5:
            self.img = self.IMGS[0]
            self.img_count = 0

        # Straight dunk while coming down
        if self.tilt <= -75:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2

        # Now we actually rotate the image
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        bird_rectangle = rotated_image.get_rect(
            center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, bird_rectangle)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 200
    VELOCITY = 5

    def __init__(self, x):
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0

        self.PIPE_BOTTOM = PIPE_IMG
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)

        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VELOCITY

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()

        top_pipe_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_pipe_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_pipe_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_pipe_offset = (self.x - bird.x, self.bottom - round(bird.y))

        top_collision_point = bird_mask.overlap(top_pipe_mask, top_pipe_offset)
        bottom_collision_point = bird_mask.overlap(
            bottom_pipe_mask, bottom_pipe_offset)

        if top_collision_point or bottom_collision_point:
            return True

        return False


class Base:
    VELOCITY = 5
    IMG = BASE_IMG
    WIDTH = IMG.get_width()

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VELOCITY
        self.x2 -= self.VELOCITY

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(win, birds, pipes, base, score):
    win.blit(BACK_IMG, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    base.draw(win)

    for bird in birds:
        bird.draw(win)
    pygame.display.update()


def eval_genome(genomes, config):
    birds = []
    nets = []
    ge = []

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        genome.fitness = 0
        ge.append(genome)

    base = Base(730)
    pipes = [Pipe(700)]

    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    score = 0

    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

                pygame.quit()
                quit()

        # Checking if the bird is behind the first pipe
        pipe_index = 0
        if len(birds) > 0:
            # Incrementing the pipe index when crossed
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_index = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness = 0.1

            # We'll check value of output. If the value increases the threshold then the bird jumps
            output = nets[x].activate((bird.y, abs(
                bird.y - pipes[pipe_index].height), abs(bird.y - pipes[pipe_index].bottom)))
            if output[0] > 0.5:
                bird.jump()

        add_pipe = False
        rem = []
        for pipe in pipes:
            for one, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[one].fitness -= 1

                    # Actually removing the bird and its properties
                    birds.pop(one)
                    nets.pop(one)
                    ge.pop(one)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        if add_pipe:
            score += 1

            # Bird crossing the pipe will get extra credit
            # These are genomes for actually alive birds
            for g in ge:
                g.fitness += 5

            pipes.append(Pipe(600))

        for r in rem:
            pipes.remove(r)

        for dead, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:

                # Removing the birds when they actually hit the ground
                birds.pop(dead)
                nets.pop(dead)
                ge.pop(dead)

        base.move()
        draw_window(win, birds, pipes, base, score)


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    winner = population.run(eval_genome, 30)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "FlappyBird-NEAT.config")
    run(config_path)
