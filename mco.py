#import numpy as np 
from  math import sqrt as sq
from pprint import pprint 
from numpy import linalg as la
from pylab import plot, subplot, axis, stem, show, figure
import os, glob, math 
import matplotlib.pyplot as plt

# definicion de funciones

# def 1: crea un txt con el nombre de los archivos 
def maketxt():
    
    path = 'Brahe-3141-f/'
    experiments = [f for f in os.listdir(path) if f.endswith('.dat')]
   
    return experiments

# def 2: carga los datos en cada archivo 
def loadData(nameFile):

    m = 38
    name = nameFile.replace('\n','')
    data = np.loadtxt("Brahe-3141-f/"+name)
    t = [ data[i][0] for i in range(m)]
    t2= [ t[i]*t[i]/2.0 for i in range(m)]
    Y = [ data[i][1] for i in range(m)]
    u = [1.0 for i in range(m)]
    T = [u,t,t2]
    Y = np.asarray(Y)
    T = np.asarray(T)
    return Y,T

# def 3: carga los datos de theta y la gravedad caluculada 
def loadData2(file):
    
    m = 1000
    data = np.loadtxt(file)
    theta = [ data[i][0] for i in range(m)]
    g = [ data[i][2] for i in range(m)]
    theta = np.asarray(theta)
    g = np.asarray(g)
    
    return theta, g

# def 4: obtiene los angulos del nombre de cada archivo
def param(nameFile):

    temp = nameFile.replace('\n','')
    temp = nameFile.replace('experiment_theta_','')
    temp = temp.replace('.dat','')
    temp = temp.replace('_phi','')
    temp = temp.split('_')
    theta = float(temp[0])
    phi = float(temp[1])
    return theta, phi
   
# def 5: realiza la descomposicion de Cholesky 
def cholesky(A):
    n = len(A)
    L = np.zeros(shape = (n,n))
    
    for i in range(n):
        for k in range(i+1):
            ksuma = sum( L[i][j] * L[k][j] for j in range(k) )

            if (i == k):
                L[i][k] = sq(abs( A[i][i] - ksuma ))

            else:
                L[i][k] = (( 1.0 / L[k][k] ) * ( A[i][k] - ksuma ))
    return np.asarray(L)

# def 6: escribe un archivo con el mensaje entregado
def write_file(dataOutput, message):
    output = open(dataOutput, "a")
    output.write(message)

# def 7: genera las matrices de datos para el ajuste del modelo con las hipotesis 1 y 2
def matrix(ang, a):
    if( a==0 ):
        M1 = [ np.cos(2.0*ang[i]) for i in range(len(ang)) ] 
        M0 = [ 1.0 for i in range(len(ang)) ]
        M = [M0, M1]
    if( a==1 ):
        M2 = [ ang[i]*ang[i] for i in range(len(ang)) ]
        M1 = [ ang[i] for i in range(len(ang)) ] 
        M0 = [ 1.0 for i in range(len(ang)) ]
        M = [M0, M1, M2]
    return np.asarray(M).T
        
# INICIO

# inicializaciones
tam = 1000
experiments = maketxt()
    
# punto 1: se realiza la estimacion de los parametros para cada archivo a traves deCholesky
for exp in experiments:
    
    Y,t = loadData(exp)
    t = t.T
    A = np.dot(t.T,t)
    b = np.dot(t.T,Y)

    # queda el sistema Ax=b y se halla L mariz triangular inferior
    L = cholesky(A)
    q = la.solve(L,b)
    x = la.solve(L.T,q)
    theta, phi = param(exp)
    y0 = x[0]
    v0 = x[1]
    g = -(x[2])
    message = str(theta) + ' ' +  str(phi) + ' ' +  str(g) + ' ' +  str(v0) + ' ' +  str(y0) + "\n"
    write_file('data.dat', message)

# punto 2: se descartan los datos corruptos y se grafica g vs theta 
theta1, g1 = loadData2('data.dat')
plt.plot(theta1, g1, 'ro')
plt.xlabel('Theta (Grad)')
plt.ylabel('G (m/s2)')
plt.title('G vs Theta sin filtrar')
plt.show()

# Se eliminan los datos corruptos
 
g = []
theta = []
for i in range(len(g1)):
    for j in range(len(g1)):
        if ( abs(theta1[i]-theta1[j])<5.0 ):
            near = j 
    if( abs(g1[i]-g1[near])<0.5 ):
        g.append(g1[i])
        theta.append(theta1[i]*( (math.pi)/180.0 ))
g = np.asarray(g)
theta = np.asarray(theta)

plt.plot(theta, g, 'ro')
plt.xlabel('Theta (rad)')
plt.ylabel('G (m/s2)')
plt.title('G vs Theta filtrado')
plt.show()
 
# punto 3: se parametriza la gravedad 
F = [] 
gMean = np.mean(g)

for i in range(len(g)):
    F.append(( 1.0 - ( g[i]/gMean )))

plt.plot(theta, F, 'ro')
plt.xlabel('Theta (rad)')
plt.ylabel('F')
plt.title('F vs Theta')
plt.grid()
plt.show()

# punto 4: se hace el ajuste de los modelos para F y theta 
'(i=0) primera hipotesis a1 + a0 cos(2theta)'
'(i=1) segunda hipotesis b0 + b1 theta + b2 theta2'

An = []
bn = []
xn = []
Mn = []
Ln = []
qn = []

for i in range(2):
    Mn.append(matrix(theta,i))
    An.append(np.dot(Mn[i].T, Mn[i]))
    bn.append(np.dot(Mn[i].T,F))
    Ln.append(cholesky(An[i]))
    qn.append(la.solve(Ln[i], bn[i]))
    xn.append(la.solve(Ln[i].T,qn[i]))
    
    if (i==0):
        a0 = xn[i][0]
        a1 = xn[i][1]
        message1 = "a0 = " + str(a0) + " a1= " +  str(a1) + "\n"  
    else:
        b0 = xn[i][0]
        b1 = xn[i][1]
        b2 = xn[i][2]
        message1 = "b0 = " + str(b0) + " b1= " +  str(b1) + " b2= " +  str(b2) + "\n"   
    write_file('fitModels.dat', message1)

# seleccion del modelo que mas se ajuste segun el R2

fhat11 = [a0 + a1*np.cos(2.0*theta[i]) for i in range(len(theta)) ]
fhat22 = [b0 + b1*theta[i] + b2*theta[i]*theta[i] for i in range(len(theta)) ]
fhat1 = np.asarray(fhat11)
fhat2 = np.asarray(fhat22)

plt.plot(theta, fhat1, 'bo')
plt.plot(theta, F,'ro')
plt.xlabel('Theta (rad)')
plt.ylabel('F')
plt.grid()
plt.title('F predicho vs Modelo 1')
plt.show()

plt.plot(theta, fhat2, 'bo')
plt.plot(theta, F,'ro')
plt.xlabel('Theta (rad)')
plt.ylabel('F')
plt.grid()
plt.title('F predicho vs Modelo 2')
plt.show()


div = sum( ( F[i]-np.mean(F) )**2 for i in range(len(F)))
r1 = sum( ( fhat1[i]-np.mean(F) )**2 for i in range(len(F))) / div
r2 = sum( ( fhat2[i]-np.mean(F) )**2 for i in range(len(F))) / div
if(r1>r2):
    mayor = "modelo 1"
else:
    mayor = "modelo 2"
message2 = "el ajuste es: " + "\n" + "r1= "+ str(r1) + "\n" + "r2= " + str(r2) + "\n" + "Donde se toma el modelo con el mayor R2, en este caso el " + mayor +"\n" 
write_file('fitModels.dat', message2)




