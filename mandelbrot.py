#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  [mandelbrot.py]
#  
#  Copyright [2015] Gabriel Hondet <gabrielhondet@gmail.com>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#

import numpy as np
import pylab
from multiprocessing import Process, Queue
import math
import numba

@numba.autojit
def mandelbrot(iterations, x, y):
    """
        renvoit le nombre d'itérations
        nécessaires a la divergence de
        z
        z diverge ssi |z| > 2
        si z ne diverge pas, renvoit 255
    """
    i = 0
    z = 0j
    c = complex(x, y)
    z = z**2 + c
    while abs(z) < 2 and i < iterations:
            z = z**2 + c
            i += 1
    nu = math.log(math.log(abs(z))/math.log(10e10))/math.log(2)
    if i == iterations:
        return 0
    else:
        return (i + 1 - nu)%(iterations/5)

@numba.autojit
def fractale(x_coor, y_coor, width_px, height_px, iterations, pile, pos):
    """
        crée la fractale
        chaque point est traité
    """
    img = np.zeros((height_px, width_px))
    x_px_size = (x_coor['max'] - x_coor['min'])/width_px
    y_px_size = (y_coor['max'] - y_coor['min'])/height_px

    for i in range(width_px):
        x_real = i*x_px_size + x_coor['min']
        for k in range(height_px):
            y_real = k*y_px_size + y_coor['min']
            img[k,i] = mandelbrot(iterations, x_real, y_real)
    
    pile.put((img, pos))

def main():
	"""
	Créer les threads, rassemble les images, sauvegarde
	On prend [-2,1] et [-1,1] en x et y pour avoir tout l'ensemble,
	le ratio étant de 3/2, il est judicieux de garder ce ration pour
	les tailles en pixels
	"""
	pile = Queue()
	cmap = 'BrBG'
	# Paramètres de l'image finale
	x_range = {'min' : -2, 'max' : 1}
	y_range = {'min' : -1, 'max' : 1}
	
	# N threads
	n_threads = 4
	assert math.sqrt(n_threads) == int(math.sqrt(n_threads)) # On s'assure d'avoir un carré parfait
	nt_dim = int(math.sqrt(n_threads))
	# Tableau : threads[i][j] : ligne i colonne j
	threads = [[None for i in range(nt_dim)] for i in range(nt_dim)]
	for i in range(nt_dim):
		y_min_th = y_range['min'] + i*(y_range['max'] - y_range['min'])/nt_dim
		y_max_th = y_range['min'] + (i + 1)*(y_range['max'] - y_range['min'])/nt_dim
		for j in range(nt_dim):
			x_min_th = x_range['min'] + j*(x_range['max'] - x_range['min'])/nt_dim
			x_max_th = x_range['min'] + (j + 1)*(x_range['max'] - x_range['min'])/nt_dim
			threads[i][j] = Process(target=fractale,
									args=({'min' : x_min_th, 'max' : x_max_th},
										  {'min' : y_min_th, 'max' : y_max_th},
										  4500, 3000, 200, pile, (i, j)))
	
	for i in range(nt_dim):
		for j in range(nt_dim):
			threads[i][j].start()
	
	# Récupération des morceaux d'images
	img_pieces_mess = [[None for i in range(nt_dim)] for i in range(nt_dim)]
	p = [[None for i in range(nt_dim)] for i in range(nt_dim)]
	for i in range(nt_dim):
		for j in range(nt_dim):
			img_pieces_mess[i][j], p[i][j] = pile.get()
	# Remise en ordre des images :
	img_pieces = [[None for i in range(nt_dim)] for i in range(nt_dim)]
	for i in range(nt_dim):
		for j in range(nt_dim):
			img_pieces[p[i][j][0]][p[i][j][1]] = img_pieces_mess[i][j]
	# Rassemblement des images : concaténation des colonnes puis lignes
	img_onecolumn = [None for i in range(nt_dim)]
	for i in range(nt_dim):
		img_onecolumn[i] = np.hstack((img_pieces[i][0], img_pieces[i][1]))
		for j in range(2, nt_dim):
			img_onecolumn[i] = np.hstack((img_onecolumn[i], img_pieces[i][j]))
	# Concaténation des lignes :
	img_tot = np.vstack((img_onecolumn[0], img_onecolumn[1]))
	for i in range(2, nt_dim):
		img_tot = np.vstack((img_tot, img_onecolumn[i]))
	
	for i in range(nt_dim):
		for j in range(nt_dim):
			threads[i][j].join()
	
	#pylab.imshow(img_tot, cmap=cmap)
	pylab.imsave('mandelbrot_normalized.png', img_tot, cmap=cmap)
	#pylab.show()

if __name__ == '__main__':
    main()
