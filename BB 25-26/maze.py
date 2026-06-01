import os
import time

# pip install pillow
from PIL import Image 
import random
import sys
import shutil
from collections import deque
import time

VISITED = (0, 0, 255)
WALKED = (255, 0, 0)
PATH = (255, 255, 255)
WALL = (0, 0, 0)

class Pixel:
	red = 0
	green = 0
	blue = 0

	def __init__(self, r, g, b):
		self.red = r
		self.green = g
		self.blue = b

	def setcolors(self, r, g, b):
		self.red = r
		self.green = g
		self.blue = b

	def getcolors(self) -> tuple:
		return (self.red, self.green, self.blue)


# klasse die alles regelt rond een jpg
class Imageclass:
	im = None # this is the PIL Image
	pad = ""

	def __init__(self, pad):
		self.source_path = pad
		try:
			imfile = Image.open(self.source_path)
			self.im = imfile.convert('RGBA')
		except:
			pass
		if self.im is None:
			sys.exit("Wrong image path or format. Exit.")

	def getwidth(self) -> int:
		b, h = self.im.size
		return b

	def getheight(self) -> int:
		b, h = self.im.size
		return h

	def getpixel(self, x, y) -> Pixel:
		# gets pixel as RGB list
		ding = self.im.getpixel((x, y))
		pixel = Pixel(ding[0], ding[1], ding[2])
		return pixel

	def ispath(self, x, y) -> bool:
		pixel = self.getpixel(x, y)
		return pixel.getcolors() == PATH

	def iswall(self, x, y) -> bool:
		pixel = self.getpixel(x, y)
		return pixel.getcolors() == WALL

	def isvisited(self, x, y) -> bool:
		pixel = self.getpixel(x, y)
		return pixel.getcolors() == VISITED

	def get_visited_states(self, state) -> list:
		# return all neighbours from current state
		x, y = state
		candidates = [
			('u', (x, y - 1)),
			('d', (x, y + 1)),
			('l', (x - 1, y)),
			('r', (x + 1, y))
		]
		result = list()
		for action, (ax, ay) in candidates:
			if 0 <= ax < self.getwidth() and 0 <= ay < self.getheight() and not self.iswall(ax, ay):
				result.append((action, (ax, ay)))
		return result

	def get_valid_states(self, state) -> list:
		# returns all possible – not visited – neighbours from current state
		x, y = state
		candidates = [
			('u', (x, y - 1)),
			('d', (x, y + 1)),
			('l', (x - 1, y)),
			('r', (x + 1, y))
		]
		result = list()
		for action, (ax, ay) in candidates:
			if self.ispath(ax, ay):
				result.append((action, (ax, ay)))
		return result

	def paintpixel(self, x: int, y: int, rgb: tuple):
		pixel = self.getpixel(x, y)
		pixel.red = rgb[0]
		pixel.green = rgb[1]
		pixel.blue = rgb[2]
		self.putpixel(pixel, x, y)

	def putpixel(self, pixel, x, y):
		# sets pixel as RBB list in Image im
		# ignore the alfa channel
		p = (self.to255(pixel.red), self.to255(pixel.green),
			self.to255(pixel.blue))
		try:
			self.im.paste(p, (x, y, x + 1, y + 1))
		except:
			sys.exit("Putting pixel back failed")

	def show(self):
		self.im.show()

	def write(self) -> str:
		try:
			self.im.save(self.source_path)
			return self.source_path
		except:
			sys.exit("Writing to file failed")

	# sets erin value to int of min 0 and max 255
	def to255(self, erin) -> int:
		return max(0, min(255, int(round(erin))))


# the node class
class Node:
	def __init__(self, state, parent, action, hn, gn):
		self.state = state
		self.parent = parent
		self.action = action
		self.hn = hn
		self.gn = gn

# the Frontier class for DFS method

# the maze class
# THIS IS WHERE THE AI LIVES
class Maze:
	def __init__(self, im: Imageclass):
		# the image
		self.im = im

		# startpoint
		self.start = (None, None)
		self.start = self.find_start()
		self.end = self.find_start()

		# check entrance and exit
		if None in self.start or None in self.end:
			sys.exit('geen start of eind gevonden')

		# counter
		self.num_explored = 0
		
		print(f'SOLVE {self.im.pad} with Gbfs. Start @ {self.start}, end @ {self.end}')


	# finds the starting point of the maze - the entrance 
	# using the self.im method ispath(x, y), which returns True if path and False if wall
	def find_start(self) -> (int | None, int | None):  # x,y
		start = None
		for y in range(0, self.im.getheight()):
			if y == 0 or y == self.im.getheight() - 1:
				for x in range(0, self.im.getwidth()):
					if self.im.ispath(x, y) and (x, y) != self.start:
						return x, y
			else:
				for x in [0, self.im.getwidth() - 1]:
					if self.im.ispath(x, y) and (x, y) != self.start:
						return x, y
		return None, None

	
	# finds the manhattan value of a given state
	def calc_manhattan(self, state: tuple):
		current_x, current_y = state
		end_x, end_y = self.end
		return abs(current_x - end_x) + abs(current_y - end_y)


	# paints a pixel in self.im as visited. 
	def visited_pixel(self, state: tuple):
		self.im.paintpixel(state[0], state[1], VISITED)

	
	# paints a pixel in self.im as walked 
	def walked_pixel(self, state: tuple):
		self.im.paintpixel(state[0], state[1], WALKED)
		
	# for reporting only
	def now(self):
		return round(time.time() * 1000)
	
	
	# write the image at the end, just for reporting
	def wrapup(self, node):
		print()
		actions = []
		while node.parent is not None:
			# walk backwards
			actions.append(node.action)
			self.walked_pixel(node.state)
			node = node.parent

		totpix = self.im.getwidth() * self.im.getheight()
		perc_expl = round(self.num_explored / totpix * 10000) / 100
		perc_walk = round(len(actions) / totpix * 10000) / 100
		self.im.write()
		print(
			f'Explored: {self.num_explored}, walked: {len(actions)} of {totpix} pixels. \n\tExplored {perc_expl}%. \n\tWalked {perc_walk}%.')	
	
		
	# solving the maze
	def solve(self):
		gestart = self.now()
		# start position, empty frontier
		start = Node(state=self.start, parent=None, action=None, hn=self.calc_manhattan(self.start), gn=0)
		# hn = manhattan calc of expected path
		# gn = real distance traveled
		
		# this is the class for the frontier
		frontier = GBfsFrontier(self.end)
		# starting point.
		frontier.add(start)

		# reporting
		print(f'explore...           \t\t', end='', flush=True)
		node = start
		while True:
			
			# THIS IS WHERE THE AI TAKES PLACE
			
			if frontier.is_empty():
				raise Exception("no solution")

			# this is where the smartness lives
			node = frontier.remove(current=node, im=self.im)

			# keeping track, DO NOT MOVE OR CHANGE
			self.num_explored += 1
			self.visited_pixel(node.state)

			# found end of maze, DO NOT MOVE OR CHANGE
			if node.state == self.end:
				self.wrapup(node)
				return

			# add new nodes to examine to the frontier
			for action, state in self.im.get_valid_states(node.state): # only contains non-visited paths and states of NEIGHBOURING nodes
				if not frontier.contains_state(state):
					child = Node(state=state, parent=node, action=action, hn=self.calc_manhattan(state), gn=node.gn+1)
					frontier.add(child)

			# NOTHING TO DO HERE, for reporting in the prompt only
			if self.num_explored % 1000 == 0:
				nps = 1000/(self.now()-gestart)
				print(f'\rexplored: [{self.num_explored}] frontier: [{len(frontier.frontier)}]  state: [{node.state}] nodes/sec: [{nps:.2f}] ', flush=True, end='')
				gestart = self.now()			


class GBfsFrontier:
	def __init__(self, end: tuple):
		# deque is a faster type of list for this context.
		# stores all candidate nodes waiting to be explored
		self.frontier = deque()
		# stores states already added
		self.visited = set() 


	def add(self, node):
		# add the node somewhere to the frontier
		self.frontier.append(node)

		# mark it as seen
		self.visited.add(node.state)


	def contains_state(self, state) -> bool:
		# check if the state is in the frontier, returns a boolean
		return state in self.visited


	def is_empty(self) -> bool:
		# checks if the frontier is empty, returns a boolean
		return len(self.frontier) == 0


	def remove(self, current: Node|None=None, im: Imageclass|None=None):
		# searches somehow for the node to remove and to continue the search with
		best = min(self.frontier, key=lambda node: node.hn)
		self.frontier.remove(best)
		return best


def maze_runner(source_name: str):
	# the name of the image file
	source_name = source_name.strip()
	target_name = f"solved-{source_name}"

	# make the output file
	try:
		shutil.copyfile(source_name, target_name)
	except:
		sys.exit(f"File {source_name} not found.")
		
	# make the pillow Image object
	im = Imageclass(target_name)
	
	# start the Maze object 
	o_maze = Maze(im)
	
	# start couning time
	nu = round(time.time() * 1000)
	o_maze.solve()
	# solved time
	ms = round(time.time() * 1000) - nu
	
	# print results
	print(f'FINISHED with GBfs {target_name} AFTER {ms} ms')
	sys.exit()


if __name__ == "__main__":
	# name to your image in same dir as this script
	image_path = "franciso-0-holes.png"
	maze_runner(image_path)

