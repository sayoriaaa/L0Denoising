import argparse
import pyvista as pv

parser = argparse.ArgumentParser(description="implementation of image smoothing via L0 gradient minimization")

#parser.add_argument('--input', default='./input/fandisk.obj', 
parser.add_argument('--input', default='./res.obj', 
help="input mesh file")

parser.add_argument('--output', default='mesh.png', 
help="output rendered image")

parser.add_argument('--show', action='store_true', 
help="create 3d interface")

args = parser.parse_args()

mesh = pv.read(args.input)

plotter = pv.Plotter(off_screen = not args.show)
plotter.add_mesh(mesh, show_edges=True, color='lightblue')
plotter.view_xy()

# plotter.screenshot(args.output)
plotter.show(screenshot=args.output)

