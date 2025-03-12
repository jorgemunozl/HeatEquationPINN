import numpy as np

# Definir la función matemática
def my_function(x):
    return  np.log(x)+0.1*x  # Ejemplo: f(x) = sin(x) * e^(-x/500)

# Generar 1000 valores aleatorios entre 0 y 1000
x_values = np.random.uniform(0, 100, 1000)

# Calcular los valores de la función
y_values = my_function(x_values)

# Redondear los valores a 4 decimales
x_values = np.round(x_values, 7)
y_values = np.round(y_values, 7)

# Apilar los valores en columnas
data = np.column_stack((x_values, y_values))

# Guardar en un archivo CSV
np.savetxt("funcion_valores.csv", data, delimiter=",", fmt="%.5f")

print("Archivo 'funcion_valores.csv' guardado exitosamente.")
