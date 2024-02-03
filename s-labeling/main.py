from queue import PriorityQueue
from numpy.random import rand
import math
import random
import numpy as np
import time
from sys import argv
from time import time



operaciones = 0

################################# clase grafo ################################################

class Grafo:

################################# metodo constructor ##########################

	def __init__(self, nodos, dirigido = True):
		self.nodos = nodos
		self.dirigido = dirigido
		self.lista_ady = {}
		for nodo in self.nodos:
			self.lista_ady[nodo] = []

		self.aristas = {}


	#metodod para añadir una arista entre los nodos
	def añadir_arista(self,nodo_fuente, nodo_destino):
		self.lista_ady[nodo_fuente].append(nodo_destino)
		
        
		if not self.dirigido:
			self.lista_ady[nodo_destino].append(nodo_fuente)



	#metodo para imprimir la lista de adyacencia

	def lista(self):
		return self.lista_ady

	def imprimir(self):
		for i in self.lista_ady:
			print(i,"",self.lista_ady[i])


	def funcion(self):
		valor = 0
		global operaciones
		for i in self.lista_ady:
			for j in self.lista_ady[i]:
				if int(i) < int(j):
					self.aristas[i +" "+ j] = min(self.etiquetas[int(i)-1],self.etiquetas[int(j)-1])
					valor  = valor + min(self.etiquetas[int(i)-1],self.etiquetas[int(j)-1])
					operaciones +=1


		
		return valor



	def eval_funcion(self,etiquetas):
		valor = 0
		global operaciones

		
		for i in self.lista_ady:
			for j in self.lista_ady[i]:
				if int(i) < int(j):
					#self.aristas[i +" "+ j] = min(self.etiquetas[int(i)-1],self.etiquetas[int(j)-1])
					valor  = valor + min(etiquetas[int(i)-1],etiquetas[int(j)-1])
					operaciones +=1


		
		return valor

################################# set values and greedys algorithms ##########################

	def set_values(self,mode):

		self.etiquetas = []

		if mode ==  'secuencial':
			for i in range(len(nodos)):
				self.etiquetas.append(i+1)

		if mode == 'aleatorio':
			memoria = []
			for i in range(len(nodos)):
				bandera = 0
				while(bandera == 0):
					num = random.randint(1, len(nodos)+1)
					if num not in memoria:
						self.etiquetas.append(num)
						memoria.append(num)
						bandera = 1

		return(self.etiquetas)

	def greedy(self):


		self.etiquetas = np.zeros(len(nodos))

		cont = 1 
		nod_grad = []

		for i in self.lista_ady:
			nod_grad.append((i,len(self.lista_ady[i])))

		nod_grad.sort(key = lambda x: x[1],reverse=True)

		for i in nod_grad:
			self.etiquetas[int(i[0])-1] = cont
			cont += 1
		return self.etiquetas

############################################################# greedy 2 ####################################################################################
	def decon(self):


		self.etiquetas = np.zeros(len(nodos))


		self.greedy()
		

		bolsa = []
		b = [0,1]
		nodes =[]
	

		for i in range(len(self.etiquetas)):

			bolado = int(np.random.choice(b,1,p = [0.5,0.5]))
			if bolado == 1:
				
				bolsa.append(self.etiquetas[i])
				nodes.append(i)
				self.etiquetas[i] = len(self.nodos)+1

		noc = []
		while bool(bolsa) != False:
			v = min(bolsa)
			bolsa.pop(bolsa.index(v))
			
			eva = []
			for i in nodes:
				if i not in noc:
					self.etiquetas[i] = v
					fun = self.funcion()
					
					self.etiquetas[i] = len(self.nodos)+1
					eva.append((fun,i))
				
			eva.sort(key = lambda x: x[0])
			x = eva[0]
			
			x = x[1]
			#ind = eva.index(x)
			self.etiquetas[x] = v
			noc.append(x)

		return self.etiquetas

########################################## evolucion diferencial ###################################


	## poblacion
	def get_pob_unfm(self,tamPob,varDes):
		global operaciones
		operaciones +=1
		pob  = []
		for i in range(tamPob): 
			pob.append(np.random.uniform(0,1, varDes))

		return pob

	## random key encoding
	def rk_fun_eval(self,pob,varDes):

		
		rk_pob =[]
		for i in pob:
			rk = np.zeros(varDes)
			tup = []
			cont = 0
			for j in i:
				tup.append((cont+1,j))
				cont+=1

						
			tup.sort(key = lambda i:i[1])
			

			cont = 0
			for j in tup:
				rk[cont] = j[0]
				cont+=1

			rk_pob.append(rk)
		
		return rk_pob

	def rk_unitario(self,i,varDes):
		
			
		rk = np.zeros(varDes)
		tup = []
		cont = 0
		for j in i:
			tup.append((cont+1,j))
			cont+=1

					
		tup.sort(key = lambda i:i[1])
		

		cont = 0
		for j in tup:
			rk[cont] = j[0]
			cont+=1
		return rk

	def rk_decode(self,x, varDes):


		y = np.random.uniform(0,1, varDes)
		y = np.sort(y)


		rk_r = np.zeros(varDes)

		for i in range(varDes):
			rk_r[int(x[i])-1] = y[i]

		return rk_r

	
	def eval_fun(self,pob):

		global 	operaciones
	
		

		evalu = [] 
		for indix in pob:
			valor = 0
			self.etiquetas = indix
			for i in self.lista_ady:
				for j in self.lista_ady[i]:
					if int(i) < int(j):
						self.aristas[i +" "+ j] = min(self.etiquetas[int(i)-1],self.etiquetas[int(j)-1])
						valor  = valor + min(self.etiquetas[int(i)-1],self.etiquetas[int(j)-1])
						operaciones +=1

			evalu.append(valor)
		
		return evalu



	## mutacion
	def mutacion(self,tamPob,pob,B,varDes):



		pobMut = []
		for i in range(tamPob):
			bolsa = []
			while len(bolsa) != 3:
				num = random.randint(0,tamPob-1)
				ban = 0
				if num not in bolsa:
					bolsa.append(num)
			
			x1 = pob[bolsa[0]]
			x2 = pob[bolsa[1]]
			x3 = pob[bolsa[2]]

			muta = x1 + B*(x2-x3)

			for i in range(varDes):
				if muta[i] < 0:
					muta[i] = 0
				if muta[i] > 1:
					muta[i] = 1

			pobMut.append(muta)

		return pobMut

	## cruza


	def cruza(self,tamPob,varDes,pob,pobMut):

		

		cruza = []

		for i in range(tamPob):

			c = np.zeros(varDes)
			m = pobMut[i]
			p = pob[i]
			num = rand(varDes)

			for j in range(varDes):

				if num[j] < 0.8:
					c[j] = p[j]
				if num[j] > 0.8:
					c[j] = m[j]

			cruza.append(c)
		
		return cruza


	## seleccion
	def seleccion(self,tamPob,evalu,evalu_cruza,pob,pobCruza):


		seleccion = []
		for i in range(tamPob):

			if evalu_cruza[i] <= evalu[i]:
				seleccion.append(pobCruza[i])
			if evalu_cruza[i] >  evalu[i]:
				seleccion.append(pob[i])

		return seleccion




	def cruza(self,tamPob,varDes,pob,pobMut):
		cruza = []

		for i in range(tamPob):

			c = np.zeros(varDes)
			m = pobMut[i]
			p = pob[i]
			num = rand(varDes)

			for j in range(varDes):

				if num[j] < 0.8:
					c[j] = p[j]
				if num[j] > 0.8:
					c[j] = m[j]

			cruza.append(c)
		
		return cruza


	## seleccion
	def seleccion(self,tamPob,evalu,evalu_cruza,pob,pobCruza):

		seleccion = []
		for i in range(tamPob):

			if evalu_cruza[i] <= evalu[i]:
				seleccion.append(pobCruza[i])
			if evalu_cruza[i] >  evalu[i]:
				seleccion.append(pob[i])

		return seleccion


########################################### evolución diferencial sin greedy ########################################
	def evo_dife(self,tamPob,varDes,B):

		pob = self.get_pob_unfm(tamPob,varDes)

		#### inicializando 3 individuos con el algoritmo greedy 2
		x = self.greedy()
		x = np.array(x)
		rk_d = self.rk_decode(x, varDes)
		pob[3] = rk_d
		
		y = self.decon()
		y = np.array(y)
		rk_d = self.rk_decode(y, varDes)
		pob[11] = rk_d

		y = self.decon()
		y = np.array(y)
		rk_d = self.rk_decode(y, varDes)
		pob[19] = rk_d
		

		rk_pob = self.rk_fun_eval(pob,varDes)
		evalu = self.eval_fun(rk_pob)

		cont = 0


		while cont!=1000:

			
			pobMut = self.mutacion(tamPob,pob,B,varDes)

			pobCruza = self.cruza(tamPob,varDes,pob,pobMut)
			rk_pobCruza = self.rk_fun_eval(pobCruza,varDes)
			evalu_cruza = self.eval_fun(rk_pobCruza)

			selecc = self.seleccion(tamPob,evalu,evalu_cruza,pob,pobCruza)
			pob = selecc

			rk_pob = self.rk_fun_eval(pob, varDes)
			evalu = self.eval_fun(rk_pob)
			
			cont +=1
		
		
		return evalu

########################################### evolución diferencial sin greedy ########################################
	def evolucion_diferencial(self,tamPob,varDes,B):

		pob = self.get_pob_unfm(tamPob,varDes)
		rk_pob = self.rk_fun_eval(pob,varDes)
		evalu = self.eval_fun(rk_pob)

		cont = 0

		while cont!=1000:

			
			pobMut = self.mutacion(tamPob,pob,B,varDes)
			pobCruza = self.cruza(tamPob,varDes,pob,pobMut)
			rk_pobCruza = self.rk_fun_eval(pobCruza,varDes)

			evalu_cruza = self.eval_fun(rk_pobCruza)


			selecc = self.seleccion(tamPob,evalu,evalu_cruza,pob,pobCruza)
			pob = selecc
			rk_pob = self.rk_fun_eval(pob, varDes)
			evalu = self.eval_fun(rk_pob)

 
			
			cont +=1
		
		
		return evalu

################################################# busqueda local ####################################
	def swap_perm(self,perm,i,j,varDes):
		swap = perm
		temp = swap[i] 
		swap[i] = swap[j]
		swap[j] = temp

		return swap

	def busqueda_local(self,pob,tamPob,varDes):
		ind = pob[random.randint(0,tamPob-1)]
		ind = grafo.rk_unitario(ind, varDes)
		ind_eval = grafo.eval_funcion(ind)
		c = 0
		while(c != 10):
			flag = 0
			for j in range(1,len(ind)):				
				x = grafo.swap_perm(ind,1,j,len(ind))
				x_eval = grafo.eval_funcion(x)
				if x_eval < ind_eval:
					flag = 1
					break
			if flag == 1:
				x_ind =x
				ind = x	
				ind_eval = x_eval	
			c +=1

		return x




########################################## lectura y creacion del grafo #############################
script, grafo, algoritmo = argv

archivo = []
nodos = []
aristas = []

##abriendo el archvio txt en modo lectura
file1 = open(grafo, "r")

## vectorizando el archivo
for i in file1: archivo.append(str(i)) 
file1.close()

## obteniendo el numero de nodos del archivo
get_val = archivo[1].split() 
num_nodos = int(get_val[1])
num_conexiones = int(get_val[2])

## vector de nodos
for i in range(num_nodos):
	nodos.append(str(i+1))

## creando el objeto grafo
grafo = Grafo(nodos,False) 

## llenado del grafo
for i in range(2,len(archivo)):
	val = (archivo[i].split())#
	grafo.añadir_arista(nodos[int(val[0])-1], nodos[int(val[1])-1])

N = 2

################################ greedy2 ###########################
if (algoritmo == 'greedy2'):

	resultados = []
	tiempo = []

	for i in range(N):
		start_time = time()
		x = np.array(grafo.decon())
		x = grafo.eval_funcion(x)
		elapsed_time = time() - start_time
		resultados.append(x)
		tiempo.append(elapsed_time)

	tiempo = np.array(tiempo)
	resultados = np.array(resultados) 
	print("\nEl algoritmo greedy 2 se ejecuto: ")
	print(str(N))
	print(" veces\n")
	print("Mejor resultado: ")
	print(str(np.min(resultados)))
	print("\nResultado promedio: ")
	print(str(np.mean(resultados)))
	print("\nDesviación estandar: ")
	print(str(np.std(resultados)))
	print("\nNúmero de operaciones: ") 
	print(str(operaciones/N))
	print("\nTiempo promedio: "), 
	print(str(np.mean(tiempo)))
	


################################ evolucion diferencial ###########################
if (algoritmo == 'evo_dif'):


	tamPob =20
	varDes =len(nodos)
	B = 20

	resultados = []
	tiempo = []

	for i in range(N):
		start_time = time()
		x = np.array(grafo.evo_dife(tamPob,varDes,B))
		x = np.min(x)
		elapsed_time = time() - start_time
		resultados.append(x)
		tiempo.append(elapsed_time)

	tiempo = np.array(tiempo)
	resultados = np.array(resultados) 
	print("\nLa evolucion diferencial se ejecuto: ")
	print(str(N))
	print(" veces\n")
	print("Mejor resultado: ")
	print(str(np.min(resultados)))
	print("\nResultado promedio: ")
	print(str(np.mean(resultados)))
	print("\nDesviación estandar: ")
	print(str(np.std(resultados)))
	print("\nNúmero de operaciones: ") 
	print(str(operaciones/N))
	print("\nTiempo promedio: "), 
	print(str(np.mean(tiempo)))














	




	








