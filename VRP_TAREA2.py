# -*- coding: utf-8 -*-
"""
CVRP con Recocido Simulado (Simulated Annealing)
Implementación de metaheurística para resolver el problema de ruteo de vehículos
con restricciones de capacidad usando Simulated Annealing.

Autores: [AGREGAR NOMBRES Y RUT DE TODOS LOS INTEGRANTES]
Fecha: Noviembre 2025
"""

import csv
import time
from datetime import datetime
import math
import random
import copy
from collections import namedtuple
import numpy as np

# Opcional: para gráficos
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_DISPONIBLE = True
except ImportError:
    MATPLOTLIB_DISPONIBLE = False
    print("[Aviso] matplotlib no disponible. No se generarán gráficos.")

# ============================
# PARÁMETROS DEL RECOCIDO SIMULADO
# ============================
# Parámetros de temperatura
TEMPERATURA_INICIAL = 1500.0  # Temperatura inicial alta para exploración
TEMPERATURA_MINIMA = 0.01  # Temperatura mínima (criterio de parada)
ALPHA = 0.995  # Factor de enfriamiento (0.95 - 0.999)

# Parámetros de iteración
ITERACIONES_POR_TEMPERATURA = 160  # Cuántas iteraciones hacer a cada temperatura
MAX_ITERACIONES_SIN_MEJORA = 1000000  # Parada adicional si no mejora

# Número de ejecuciones para estadísticas
NUMERO_EJECUCIONES = 10

# Semilla aleatoria para reproducibilidad (comentar para resultados aleatorios)
SEMILLA_ALEATORIA = 42

# Configuración de visualización
MOSTRAR_GRAFICO_FINAL = True  # Mostrar gráfico de rutas al final
MOSTRAR_CONVERGENCIA = True  # Mostrar gráfico de convergencia del algoritmo

# ============================
# ESTRUCTURAS DE DATOS
# ============================
Nodo = namedtuple("Nodo", ["id", "x", "y", "demanda"])


class InstanciaCVRP:
    """Contiene toda la información de una instancia CVRP"""
    
    def __init__(self, nombre, nodos, capacidad_vehiculo, num_vehiculos=None):
        self.nombre = nombre
        self.nodos = nodos  # Lista de objetos Nodo
        self.capacidad_vehiculo = capacidad_vehiculo
        self.deposito = nodos[0]  # Siempre el primer nodo
        self.clientes = [n for n in nodos if n.id != self.deposito.id]
        
        # Calcular número mínimo de vehículos necesarios
        demanda_total = sum(c.demanda for c in self.clientes)
        self.num_vehiculos_min = math.ceil(demanda_total / capacidad_vehiculo)
        
        # Usar número especificado o el mínimo necesario
        if num_vehiculos is None:
            self.num_vehiculos = self.num_vehiculos_min
        else:
            self.num_vehiculos = max(num_vehiculos, self.num_vehiculos_min)
        
        # Calcular matriz de distancias
        self.matriz_distancia = self._calcular_matriz_distancias()
    
    def _calcular_matriz_distancias(self):
        """Calcula la matriz de distancias euclidianas entre todos los nodos"""
        n = len(self.nodos)
        matriz = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = self.nodos[i].x - self.nodos[j].x
                    dy = self.nodos[i].y - self.nodos[j].y
                    matriz[i][j] = math.hypot(dx, dy)
        
        return matriz
    
    def distancia(self, nodo_i, nodo_j):
        """Retorna la distancia entre dos nodos por sus IDs"""
        return self.matriz_distancia[nodo_i][nodo_j]


class Solucion:
    """Representa una solución del CVRP (conjunto de rutas)"""
    
    def __init__(self, instancia, rutas=None):
        self.instancia = instancia
        self.rutas = rutas if rutas is not None else [[] for _ in range(instancia.num_vehiculos)]
        self._costo_total = None
        self._cargas = None
    
    def copiar(self):
        """Crea una copia profunda de la solución"""
        return Solucion(
            self.instancia,
            [ruta[:] for ruta in self.rutas]  # Copiar cada lista de ruta
        )
    
    def calcular_costo_total(self):
        """Calcula el costo total de todas las rutas"""
        costo = 0.0
        deposito_id = self.instancia.deposito.id
        
        for ruta in self.rutas:
            if not ruta:  # Ruta vacía
                continue
            
            # Depósito -> primer cliente
            costo += self.instancia.distancia(deposito_id, ruta[0])
            
            # Entre clientes consecutivos
            for i in range(len(ruta) - 1):
                costo += self.instancia.distancia(ruta[i], ruta[i + 1])
            
            # Último cliente -> depósito
            costo += self.instancia.distancia(ruta[-1], deposito_id)
        
        self._costo_total = costo
        return costo
    
    def calcular_cargas(self):
        """Calcula la carga de cada vehículo"""
        cargas = []
        demandas = {n.id: n.demanda for n in self.instancia.nodos}
        
        for ruta in self.rutas:
            carga = sum(demandas[cliente_id] for cliente_id in ruta)
            cargas.append(carga)
        
        self._cargas = cargas
        return cargas
    
    def es_factible(self):
        """Verifica si la solución respeta las restricciones de capacidad"""
        cargas = self.calcular_cargas()
        capacidad = self.instancia.capacidad_vehiculo
        return all(carga <= capacidad for carga in cargas)
    
    def get_costo(self):
        """Retorna el costo (calculándolo si no está en caché)"""
        if self._costo_total is None:
            self.calcular_costo_total()
        return self._costo_total
    
    def invalidar_cache(self):
        """Invalida el caché de costo y cargas tras modificar rutas"""
        self._costo_total = None
        self._cargas = None
    
    def __str__(self):
        """Representación en string de la solución"""
        deposito = self.instancia.deposito.id
        lineas = [f"Solución - Costo total: {self.get_costo():.2f}"]
        cargas = self.calcular_cargas()
        
        for i, ruta in enumerate(self.rutas):
            if ruta:
                ruta_str = f"{deposito} -> " + " -> ".join(map(str, ruta)) + f" -> {deposito}"
                lineas.append(f"  Vehículo {i}: {ruta_str} (Carga: {cargas[i]}/{self.instancia.capacidad_vehiculo})")
            else:
                lineas.append(f"  Vehículo {i}: No utilizado")
        
        return "\n".join(lineas)


# ============================
# CLASE PARA REGISTRO DE RESULTADOS
# ============================
class LogResultados:
    """Clase para registrar y generar tabla de resultados de múltiples ejecuciones"""
    
    def __init__(self, nombre_instancia):
        self.nombre_instancia = nombre_instancia
        self.resultados = {}  # Diccionario por instancia
    
    def agregar_ejecucion(self, instancia, numero_ejecucion, costo):
        """Agrega el resultado de una ejecución"""
        if instancia not in self.resultados:
            self.resultados[instancia] = []
        self.resultados[instancia].append({
            'ejecucion': numero_ejecucion,
            'costo': costo
        })
    
    def calcular_estadisticas(self, instancia):
        """Calcula media y desviación estándar para una instancia"""
        if instancia not in self.resultados or len(self.resultados[instancia]) == 0:
            return None, None
        
        costos = [r['costo'] for r in self.resultados[instancia]]
        media = np.mean(costos)
        desviacion = np.std(costos, ddof=1)  # Desviación estándar muestral
        
        return media, desviacion
    
    def generar_tabla(self):
        """Genera la tabla de resultados en formato texto"""
        print("\n" + "="*80)
        print(f"TABLA DE RESULTADOS - {self.nombre_instancia}")
        print("="*80)
        
        # Obtener todas las instancias
        instancias = sorted(self.resultados.keys())
        
        if not instancias:
            print("No hay resultados para mostrar.")
            return
        
        # Encabezado
        header = "ejecución |"
        for inst in instancias:
            header += f" {inst:^12} |"
        print(header)
        print("-" * len(header))
        
        # Determinar el número máximo de ejecuciones
        max_ejecuciones = max(len(self.resultados[inst]) for inst in instancias)
        
        # Filas de datos
        for i in range(max_ejecuciones):
            fila = f"    {i+1:2d}    |"
            for inst in instancias:
                if i < len(self.resultados[inst]):
                    costo = self.resultados[inst][i]['costo']
                    fila += f" {costo:12.2f} |"
                else:
                    fila += f" {'-':^12} |"
            print(fila)
        
        # Separador antes de estadísticas
        print("-" * len(header))
        
        # Fila de media
        fila_media = "   media   |"
        for inst in instancias:
            media, _ = self.calcular_estadisticas(inst)
            if media is not None:
                fila_media += f" {media:12.2f} |"
            else:
                fila_media += f" {'-':^12} |"
        print(fila_media)
        
        # Fila de desviación estándar
        fila_desv = "desviación |"
        fila_desv += "\n estándar  |"
        for inst in instancias:
            _, desviacion = self.calcular_estadisticas(inst)
            if desviacion is not None:
                fila_desv += f" {desviacion:12.2f} |"
            else:
                fila_desv += f" {'-':^12} |"
        print(fila_desv)
        
        print("="*80 + "\n")
    
    def exportar_csv(self, nombre_archivo="resultados.csv"):
        """Exporta los resultados a un archivo CSV"""
        instancias = sorted(self.resultados.keys())
        
        if not instancias:
            print("No hay resultados para exportar.")
            return
        
        with open(nombre_archivo, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Encabezado
            header = ['ejecución'] + instancias
            writer.writerow(header)
            
            # Determinar el número máximo de ejecuciones
            max_ejecuciones = max(len(self.resultados[inst]) for inst in instancias)
            
            # Escribir datos
            for i in range(max_ejecuciones):
                fila = [i + 1]
                for inst in instancias:
                    if i < len(self.resultados[inst]):
                        fila.append(f"{self.resultados[inst][i]['costo']:.2f}")
                    else:
                        fila.append('')
                writer.writerow(fila)
            
            # Fila vacía
            writer.writerow([])
            
            # Estadísticas
            fila_media = ['media']
            fila_desv = ['desviación estándar']
            
            for inst in instancias:
                media, desviacion = self.calcular_estadisticas(inst)
                fila_media.append(f"{media:.2f}" if media is not None else '')
                fila_desv.append(f"{desviacion:.2f}" if desviacion is not None else '')
            
            writer.writerow(fila_media)
            writer.writerow(fila_desv)
        
        print(f"Resultados exportados a: {nombre_archivo}")


# ============================
# LECTURA DE ARCHIVOS .VRP
# ============================
def leer_archivo_vrp(nombre_archivo):
    """
    Lee un archivo en formato TSPLIB/CVRPLIB (.vrp)
    
    Args:
        nombre_archivo: ruta al archivo .vrp
    
    Returns:
        InstanciaCVRP con los datos cargados
    """
    with open(nombre_archivo, 'r') as f:
        lineas = f.readlines()
    
    # Variables para almacenar datos
    nombre = ""
    dimension = 0
    capacidad = 0
    coordenadas = {}
    demandas = {}
    
    # Parsear el archivo
    seccion_actual = None
    
    for linea in lineas:
        linea = linea.strip()
        
        if not linea or linea == 'EOF':
            continue
        
        # Leer metadatos
        if linea.startswith('NAME'):
            nombre = linea.split(':')[1].strip()
        elif linea.startswith('DIMENSION'):
            dimension = int(linea.split(':')[1].strip())
        elif linea.startswith('CAPACITY'):
            capacidad = int(linea.split(':')[1].strip())
        
        # Detectar secciones
        elif linea == 'NODE_COORD_SECTION':
            seccion_actual = 'coords'
        elif linea == 'DEMAND_SECTION':
            seccion_actual = 'demands'
        elif linea == 'DEPOT_SECTION':
            seccion_actual = 'depot'
        
        # Leer datos de cada sección
        elif seccion_actual == 'coords':
            partes = linea.split()
            if len(partes) == 3:
                nodo_id = int(partes[0])
                x = float(partes[1])
                y = float(partes[2])
                coordenadas[nodo_id] = (x, y)
        
        elif seccion_actual == 'demands':
            partes = linea.split()
            if len(partes) == 2:
                nodo_id = int(partes[0])
                demanda = int(partes[1])
                demandas[nodo_id] = demanda
    
    # Crear objetos Nodo (reindexar desde 0)
    nodos = []
    ids_ordenados = sorted(coordenadas.keys())
    
    for nuevo_id, viejo_id in enumerate(ids_ordenados):
        x, y = coordenadas[viejo_id]
        demanda = demandas.get(viejo_id, 0)
        nodos.append(Nodo(nuevo_id, x, y, demanda))
    
    # Crear instancia
    instancia = InstanciaCVRP(nombre, nodos, capacidad)
    
    return instancia


# ============================
# GENERACIÓN DE SOLUCIÓN INICIAL
# ============================
def generar_solucion_inicial_voraz(instancia):
    """
    Genera una solución inicial usando heurística del vecino más cercano
    Construye rutas secuencialmente, agregando el cliente no visitado más cercano
    """
    clientes_no_asignados = set(c.id for c in instancia.clientes)
    rutas = [[] for _ in range(instancia.num_vehiculos)]
    cargas = [0] * instancia.num_vehiculos
    vehiculo_actual = 0
    
    deposito_id = instancia.deposito.id
    demandas = {n.id: n.demanda for n in instancia.nodos}
    
    while clientes_no_asignados:
        # Determinar desde dónde buscar el más cercano
        if rutas[vehiculo_actual]:
            ultimo_nodo = rutas[vehiculo_actual][-1]
        else:
            ultimo_nodo = deposito_id
        
        # Buscar cliente más cercano que quepa en el vehículo actual
        mejor_cliente = None
        mejor_distancia = float('inf')
        
        for cliente_id in clientes_no_asignados:
            demanda_cliente = demandas[cliente_id]
            
            # Verificar si cabe en el vehículo actual
            if cargas[vehiculo_actual] + demanda_cliente <= instancia.capacidad_vehiculo:
                distancia = instancia.distancia(ultimo_nodo, cliente_id)
                if distancia < mejor_distancia:
                    mejor_distancia = distancia
                    mejor_cliente = cliente_id
        
        # Si encontramos un cliente que cabe, agregarlo
        if mejor_cliente is not None:
            rutas[vehiculo_actual].append(mejor_cliente)
            cargas[vehiculo_actual] += demandas[mejor_cliente]
            clientes_no_asignados.remove(mejor_cliente)
        else:
            # Cambiar al siguiente vehículo
            vehiculo_actual += 1
            
            # Si nos quedamos sin vehículos, crear más vehículos
            if vehiculo_actual >= instancia.num_vehiculos:
                instancia.num_vehiculos += 1
                rutas.append([])
                cargas.append(0)
                print(f"[INFO] Agregando vehículo adicional. Total vehículos: {instancia.num_vehiculos}")
    
    solucion = Solucion(instancia, rutas)
    
    # Verificar que la solución sea factible
    if not solucion.es_factible():
        print("[ERROR CRÍTICO] No se pudo generar una solución inicial factible")
        print("Verificando cargas:")
        cargas = solucion.calcular_cargas()
        for i, carga in enumerate(cargas):
            if carga > instancia.capacidad_vehiculo:
                print(f"  Vehículo {i}: Carga {carga} > Capacidad {instancia.capacidad_vehiculo}")
    
    return solucion


# ============================
# OPERADORES DE VECINDAD
# ============================
def swap_intra_ruta(solucion):
    """Operador 2-opt: intercambia dos clientes dentro de la misma ruta"""
    nueva_solucion = solucion.copiar()
    
    # Seleccionar ruta no vacía aleatoriamente
    rutas_no_vacias = [i for i, r in enumerate(nueva_solucion.rutas) if len(r) >= 2]
    
    if not rutas_no_vacias:
        return nueva_solucion
    
    idx_ruta = random.choice(rutas_no_vacias)
    ruta = nueva_solucion.rutas[idx_ruta]
    
    # Seleccionar dos posiciones distintas
    i, j = random.sample(range(len(ruta)), 2)
    
    # Intercambiar
    ruta[i], ruta[j] = ruta[j], ruta[i]
    
    nueva_solucion.invalidar_cache()
    return nueva_solucion


def swap_inter_ruta(solucion):
    """
    Intercambia un cliente entre dos rutas distintas
    Verifica que se respete la capacidad
    """
    nueva_solucion = solucion.copiar()
    instancia = solucion.instancia
    
    # Seleccionar dos rutas no vacías distintas
    rutas_no_vacias = [i for i, r in enumerate(nueva_solucion.rutas) if len(r) > 0]
    
    if len(rutas_no_vacias) < 2:
        return nueva_solucion
    
    idx_ruta1, idx_ruta2 = random.sample(rutas_no_vacias, 2)
    
    # Seleccionar un cliente de cada ruta
    pos1 = random.randint(0, len(nueva_solucion.rutas[idx_ruta1]) - 1)
    pos2 = random.randint(0, len(nueva_solucion.rutas[idx_ruta2]) - 1)
    
    cliente1 = nueva_solucion.rutas[idx_ruta1][pos1]
    cliente2 = nueva_solucion.rutas[idx_ruta2][pos2]
    
    # Verificar factibilidad del intercambio
    demandas = {n.id: n.demanda for n in instancia.nodos}
    
    # Calcular cargas actuales
    carga1 = sum(demandas[c] for c in nueva_solucion.rutas[idx_ruta1])
    carga2 = sum(demandas[c] for c in nueva_solucion.rutas[idx_ruta2])
    
    # Calcular cargas después del swap
    nueva_carga1 = carga1 - demandas[cliente1] + demandas[cliente2]
    nueva_carga2 = carga2 - demandas[cliente2] + demandas[cliente1]
    
    # Solo hacer el swap si es factible
    if (nueva_carga1 <= instancia.capacidad_vehiculo and
        nueva_carga2 <= instancia.capacidad_vehiculo):
        nueva_solucion.rutas[idx_ruta1][pos1] = cliente2
        nueva_solucion.rutas[idx_ruta2][pos2] = cliente1
    
    nueva_solucion.invalidar_cache()
    return nueva_solucion


def relocate(solucion):
    """
    Mueve un cliente de una ruta a otra
    Verifica que se respete la capacidad de la ruta destino
    """
    nueva_solucion = solucion.copiar()
    instancia = solucion.instancia
    
    # Seleccionar ruta origen no vacía
    rutas_no_vacias = [i for i, r in enumerate(nueva_solucion.rutas) if len(r) > 0]
    
    if not rutas_no_vacias:
        return nueva_solucion
    
    idx_origen = random.choice(rutas_no_vacias)
    ruta_origen = nueva_solucion.rutas[idx_origen]
    
    # Seleccionar cliente a mover
    pos = random.randint(0, len(ruta_origen) - 1)
    cliente = ruta_origen[pos]
    
    # Seleccionar ruta destino diferente
    rutas_disponibles = [i for i in range(instancia.num_vehiculos) if i != idx_origen]
    
    if not rutas_disponibles:
        return nueva_solucion
    
    idx_destino = random.choice(rutas_disponibles)
    
    # Verificar factibilidad
    demandas = {n.id: n.demanda for n in instancia.nodos}
    carga_destino = sum(demandas[c] for c in nueva_solucion.rutas[idx_destino])
    
    if carga_destino + demandas[cliente] <= instancia.capacidad_vehiculo:
        # Realizar el movimiento
        ruta_origen.pop(pos)
        
        # Insertar en posición aleatoria de la ruta destino
        if nueva_solucion.rutas[idx_destino]:
            pos_insercion = random.randint(0, len(nueva_solucion.rutas[idx_destino]))
        else:
            pos_insercion = 0
        
        nueva_solucion.rutas[idx_destino].insert(pos_insercion, cliente)
    
    nueva_solucion.invalidar_cache()
    return nueva_solucion


def reverse_subruta(solucion):
    """Operador 2-opt: invierte un segmento de una ruta"""
    nueva_solucion = solucion.copiar()
    
    # Seleccionar ruta no vacía con al menos 2 elementos
    rutas_validas = [i for i, r in enumerate(nueva_solucion.rutas) if len(r) >= 2]
    
    if not rutas_validas:
        return nueva_solucion
    
    idx_ruta = random.choice(rutas_validas)
    ruta = nueva_solucion.rutas[idx_ruta]
    
    # Seleccionar dos puntos de corte
    i, j = sorted(random.sample(range(len(ruta) + 1), 2))
    
    # Invertir el segmento entre i y j
    ruta[i:j] = reversed(ruta[i:j])
    
    nueva_solucion.invalidar_cache()
    return nueva_solucion

def two_opt_inter_route(rutas, demandas, capacidad):
    import random
    if len(rutas) < 2:
        return None

    # Seleccionar dos rutas distintas
    r1, r2 = random.sample(range(len(rutas)), 2)

    ruta1 = rutas[r1]
    ruta2 = rutas[r2]

    if len(ruta1) < 2 or len(ruta2) < 2:
        return None

    # Cortes aleatorios dentro de ambas rutas
    i = random.randint(1, len(ruta1) - 1)
    j = random.randint(1, len(ruta2) - 1)

    # Crear nuevas rutas intercambiando segmentos
    nueva1 = ruta1[:i] + ruta2[j:]
    nueva2 = ruta2[:j] + ruta1[i:]

    # Validar capacidad
    demanda1 = sum(demandas[c] for c in nueva1)
    demanda2 = sum(demandas[c] for c in nueva2)

    if demanda1 > capacidad or demanda2 > capacidad:
        return None

    nuevas_rutas = rutas.copy()
    nuevas_rutas[r1] = nueva1
    nuevas_rutas[r2] = nueva2

    return nuevas_rutas

def generar_vecino(solucion):
    """Genera una solución vecina aplicando un operador aleatorio"""
    operadores = [
        swap_intra_ruta,
        swap_inter_ruta,
        relocate,
        reverse_subruta,
        two_opt_inter_route
    ]
    
    # Seleccionar operador aleatorio
    operador = random.choice(operadores)
    return operador(solucion)


# ============================
# RECOCIDO SIMULADO
# ============================
def criterio_metropolis(costo_actual, costo_nuevo, temperatura):
    """
    Decide si aceptar una nueva solución según el criterio de Metropolis
    - Si el nuevo costo es mejor (menor), siempre acepta
    - Si el nuevo costo es peor (mayor), acepta con probabilidad exp(-delta/T)
    """
    if costo_nuevo < costo_actual:
        return True
    
    if temperatura <= 0:
        return False
    
    delta = costo_nuevo - costo_actual
    probabilidad = math.exp(-delta / temperatura)
    return random.random() < probabilidad


def recocido_simulado(instancia, solucion_inicial=None, verbose=True):
    """
    Algoritmo de Recocido Simulado para CVRP
    
    Args:
        instancia: InstanciaCVRP a resolver
        solucion_inicial: Solución inicial (si es None, genera una voraz)
        verbose: si True, imprime progreso
    
    Returns:
        Mejor solución encontrada, historial de convergencia
    """
    inicio = time.time()
    
    # Generar solución inicial si no se proporciona
    if solucion_inicial is None:
        if verbose:
            print("Generando solución inicial voraz...")
        solucion_inicial = generar_solucion_inicial_voraz(instancia)
    
    # Verificar factibilidad inicial y rechazar si no es válida
    if not solucion_inicial.es_factible():
        print("[ERROR CRÍTICO] La solución inicial no es factible.")
        print("No se puede continuar con el recocido simulado.")
        cargas = solucion_inicial.calcular_cargas()
        for i, carga in enumerate(cargas):
            if carga > instancia.capacidad_vehiculo:
                print(f"  Vehículo {i}: Carga {carga} > Capacidad {instancia.capacidad_vehiculo}")
        return None, None
    
    # Inicializar
    solucion_actual = solucion_inicial.copiar()
    costo_actual = solucion_actual.get_costo()
    
    mejor_solucion = solucion_actual.copiar()
    mejor_costo = costo_actual
    
    # Historial para gráfico de convergencia
    historial_mejor_costo = [mejor_costo]
    historial_temperatura = [TEMPERATURA_INICIAL]
    historial_iteracion = [0]
    
    temperatura = TEMPERATURA_INICIAL
    iteracion = 0
    iteraciones_sin_mejora = 0
    
    if verbose:
        print(f"\n=== RECOCIDO SIMULADO ===")
        print(f"Instancia: {instancia.nombre}")
        print(f"Clientes: {len(instancia.clientes)}")
        print(f"Vehículos: {instancia.num_vehiculos}")
        print(f"Capacidad: {instancia.capacidad_vehiculo}")
        print(f"Costo inicial: {costo_actual:.2f}")
        print(f"T_inicial: {TEMPERATURA_INICIAL}, T_min: {TEMPERATURA_MINIMA}, alpha: {ALPHA}")
        print(f"\nIniciando búsqueda...\n")
    
    # Bucle principal
    while temperatura > TEMPERATURA_MINIMA and iteraciones_sin_mejora < MAX_ITERACIONES_SIN_MEJORA:
        for _ in range(ITERACIONES_POR_TEMPERATURA):
            iteracion += 1
            iteraciones_sin_mejora += 1
            
            # Generar solución vecina
            solucion_vecina = generar_vecino(solucion_actual)
            
            # Verificar factibilidad ANTES de calcular costo
            if not solucion_vecina.es_factible():
                continue
            
            costo_vecino = solucion_vecina.get_costo()
            
            # Criterio de aceptación
            if criterio_metropolis(costo_actual, costo_vecino, temperatura):
                solucion_actual = solucion_vecina
                costo_actual = costo_vecino
                
                # Actualizar mejor solución si corresponde
                if costo_actual < mejor_costo:
                    mejor_solucion = solucion_actual.copiar()
                    mejor_costo = costo_actual
                    iteraciones_sin_mejora = 0
                    
                    if verbose:
                        print(f"Iter {iteracion:6d} | T={temperatura:8.2f} | Mejor costo: {mejor_costo:.2f} ✓")
        
        # Registrar progreso
        historial_mejor_costo.append(mejor_costo)
        historial_temperatura.append(temperatura)
        historial_iteracion.append(iteracion)
        
        # Enfriar temperatura
        temperatura *= ALPHA
    
    tiempo_total = time.time() - inicio
    
    # Verificación final de factibilidad
    if not mejor_solucion.es_factible():
        print("[ERROR] La mejor solución encontrada no es factible. Esto no debería ocurrir.")
    
    if verbose:
        print(f"\n=== RESULTADO FINAL ===")
        print(f"Costo inicial: {solucion_inicial.get_costo():.2f}")
        print(f"Mejor costo encontrado: {mejor_costo:.2f}")
        mejora = ((solucion_inicial.get_costo() - mejor_costo) / solucion_inicial.get_costo()) * 100
        print(f"Mejora: {mejora:.2f}%")
        print(f"Iteraciones totales: {iteracion}")
        print(f"Tiempo: {tiempo_total:.2f} segundos")
        print(f"\n{mejor_solucion}")
    
    # Historial de convergencia
    historial = {
        'iteraciones': historial_iteracion,
        'mejor_costo': historial_mejor_costo,
        'temperatura': historial_temperatura,
        'tiempo_total': tiempo_total,
        'iteraciones_totales': iteracion
    }
    
    return mejor_solucion, historial


# ============================
# VISUALIZACIÓN
# ============================
def graficar_solucion(solucion, titulo="Solución CVRP"):
    """Grafica las rutas de la solución"""
    if not MATPLOTLIB_DISPONIBLE:
        print("[Aviso] matplotlib no disponible. No se puede graficar.")
        return
    
    instancia = solucion.instancia
    coords = {n.id: (n.x, n.y) for n in instancia.nodos}
    deposito = instancia.deposito.id
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Colores para cada vehículo - usar paleta con más colores
    colores = plt.cm.tab20(range(len(solucion.rutas)))
    
    # Dibujar rutas
    rutas_dibujadas = 0
    for i, ruta in enumerate(solucion.rutas):
        if not ruta:
            continue
        
        # Construir ruta completa: depósito -> clientes -> depósito
        ruta_completa = [deposito] + ruta + [deposito]
        xs = [coords[nodo][0] for nodo in ruta_completa]
        ys = [coords[nodo][1] for nodo in ruta_completa]
        
        ax.plot(xs, ys, 'o-', color=colores[i], linewidth=2,
                markersize=6, label=f'Vehículo {i}')
        rutas_dibujadas += 1
    
    # Marcar depósito
    ax.plot(coords[deposito][0], coords[deposito][1], 's',
            color='red', markersize=15, label='Depósito', zorder=5)
    
    # Mostrar leyenda fuera del área de graficado
    if rutas_dibujadas > 15:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                  ncol=2, fontsize=8, framealpha=0.9)
    else:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                  fontsize=9, framealpha=0.9)
    
    ax.set_title(f"{titulo}\nCosto total: {solucion.get_costo():.2f}", fontsize=12)
    ax.set_xlabel("Coordenada X", fontsize=10)
    ax.set_ylabel("Coordenada Y", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.show()


def graficar_convergencia(historial, titulo="Convergencia del Recocido Simulado"):
    """Grafica la convergencia del algoritmo"""
    if not MATPLOTLIB_DISPONIBLE:
        print("[Aviso] matplotlib no disponible. No se puede graficar.")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Gráfico de mejor costo
    ax1.plot(historial['iteraciones'], historial['mejor_costo'],
             'b-', linewidth=2)
    ax1.set_xlabel('Iteración')
    ax1.set_ylabel('Mejor Costo')
    ax1.set_title('Evolución del Mejor Costo')
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de temperatura
    ax2.plot(historial['iteraciones'], historial['temperatura'],
             'r-', linewidth=2)
    ax2.set_xlabel('Iteración')
    ax2.set_ylabel('Temperatura')
    ax2.set_title('Esquema de Enfriamiento')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(titulo, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================
# FUNCIÓN PRINCIPAL
# ============================
def main():
    """Función principal para ejecutar el recocido simulado en las 3 instancias"""
    print("="*60)
    print("     CVRP - RECOCIDO SIMULADO")
    print("="*60)
    
    # Archivos de instancias a resolver
    archivos_instancias = [
        "Facil.vrp",
        "Medio.vrp",
        "Dificil.vrp"
    ]
    
    # Crear registro de resultados global
    log_global = LogResultados("Todas las Instancias")
    
    resultados_por_instancia = {}
    
    for i, archivo in enumerate(archivos_instancias, 1):
        print(f"\n{'='*60}")
        print(f"  INSTANCIA {i}: {archivo}")
        print(f"{'='*60}")
        
        try:
            # Leer instancia
            instancia = leer_archivo_vrp(archivo)
            
            # Ejecutar múltiples veces
            print(f"\nEjecutando {NUMERO_EJECUCIONES} ejecuciones...")
            
            nombre_corto = archivo.replace('.vrp', '')
            resultados_por_instancia[nombre_corto] = []
            
            for ejecucion in range(1, NUMERO_EJECUCIONES + 1):
                print(f"\n--- Ejecución {ejecucion}/{NUMERO_EJECUCIONES} ---")
                
                # Cambiar semilla para cada ejecución
                random.seed(SEMILLA_ALEATORIA + ejecucion)
                
                # Ejecutar recocido simulado
                mejor_solucion, historial = recocido_simulado(instancia, verbose=False)
                
                # Verificar que se obtuvo una solución válida
                if mejor_solucion is None:
                    print(f"[ERROR] No se pudo resolver la ejecución {ejecucion}")
                    continue
                
                costo_final = mejor_solucion.get_costo()
                print(f"Costo obtenido: {costo_final:.2f}")
                
                # Registrar resultado
                log_global.agregar_ejecucion(nombre_corto, ejecucion, costo_final)
                resultados_por_instancia[nombre_corto].append({
                    'solucion': mejor_solucion,
                    'historial': historial,
                    'costo': costo_final
                })
            
            # Mostrar mejor solución de esta instancia
            if resultados_por_instancia[nombre_corto]:
                mejor_idx = min(range(len(resultados_por_instancia[nombre_corto])),
                               key=lambda x: resultados_por_instancia[nombre_corto][x]['costo'])
                
                mejor_resultado = resultados_por_instancia[nombre_corto][mejor_idx]
                
                print(f"\n=== MEJOR SOLUCIÓN DE {nombre_corto} ===")
                print(f"Ejecución: {mejor_idx + 1}")
                print(mejor_resultado['solucion'])
                
                # Graficar mejor solución
                if MOSTRAR_GRAFICO_FINAL:
                    graficar_solucion(mejor_resultado['solucion'],
                                    titulo=f"Mejor Solución - {nombre_corto}")
                
                # Graficar convergencia de la mejor ejecución
                if MOSTRAR_CONVERGENCIA:
                    graficar_convergencia(mejor_resultado['historial'],
                                        titulo=f"Convergencia - {nombre_corto} (Ejecución {mejor_idx + 1})")
        
        except FileNotFoundError:
            print(f"[ERROR] No se encontró el archivo '{archivo}'")
            print(f"Asegúrate de que el archivo esté en el mismo directorio que este script.")
            continue
        
        except Exception as e:
            print(f"[ERROR] Error al procesar '{archivo}': {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # ============================
    # GENERAR TABLA DE RESULTADOS
    # ============================
    print("\n" + "="*60)
    print("     GENERANDO TABLA DE RESULTADOS")
    print("="*60)
    
    # Mostrar tabla en consola
    log_global.generar_tabla()
    
    # Exportar a CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_csv = f"resultados_cvrp_{timestamp}.csv"
    log_global.exportar_csv(nombre_csv)
    
    # ============================
    # RESUMEN FINAL
    # ============================
    print(f"\n{'='*60}")
    print("     RESUMEN FINAL DE RESULTADOS")
    print(f"{'='*60}\n")
    
    for nombre_inst, resultados in resultados_por_instancia.items():
        if resultados:
            costos = [r['costo'] for r in resultados]
            print(f"{nombre_inst}:")
            print(f"  Mejor costo: {min(costos):.2f}")
            print(f"  Peor costo: {max(costos):.2f}")
            print(f"  Media: {np.mean(costos):.2f}")
            print(f"  Desviación estándar: {np.std(costos, ddof=1):.2f}")
            print()


if __name__ == "__main__":
    main()
