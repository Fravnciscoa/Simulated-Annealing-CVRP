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
        
        # MEJORA: Pre-calcular diccionario de demandas para acceso rápido
        self.demandas_dict = {n.id: n.demanda for n in self.nodos}
    
    def _calcular_matriz_distancias(self):
        """Calcula la matriz de distancias euclidianas entre todos los nodos"""
        n = len(self.nodos)
        matriz = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):  # MEJORA: Solo calcular mitad de la matriz (simétrica)
                dx = self.nodos[i].x - self.nodos[j].x
                dy = self.nodos[i].y - self.nodos[j].y
                dist = math.hypot(dx, dy)
                matriz[i][j] = dist
                matriz[j][i] = dist  # Simetría
        
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
        nueva = Solucion(self.instancia, [ruta[:] for ruta in self.rutas])
        nueva._costo_total = self._costo_total  # MEJORA: Copiar caché si existe
        nueva._cargas = self._cargas[:] if self._cargas else None
        return nueva
    
    def calcular_costo_total(self):
        """Calcula el costo total de todas las rutas"""
        costo = 0.0
        deposito_id = self.instancia.deposito.id
        distancia_func = self.instancia.distancia  # MEJORA: Cache de función
        
        for ruta in self.rutas:
            if not ruta:  # Ruta vacía
                continue
            
            # Depósito -> primer cliente
            costo += distancia_func(deposito_id, ruta[0])
            
            # Entre clientes consecutivos
            for i in range(len(ruta) - 1):
                costo += distancia_func(ruta[i], ruta[i + 1])
            
            # Último cliente -> depósito
            costo += distancia_func(ruta[-1], deposito_id)
        
        self._costo_total = costo
        return costo
    
    def calcular_cargas(self):
        """Calcula la carga de cada vehículo"""
        demandas = self.instancia.demandas_dict  # MEJORA: Usar diccionario pre-calculado
        cargas = [sum(demandas[cliente_id] for cliente_id in ruta) for ruta in self.rutas]
        self._cargas = cargas
        return cargas
    
    def es_factible(self):
        """Verifica si la solución respeta las restricciones de capacidad"""
        if self._cargas is None:
            self.calcular_cargas()
        capacidad = self.instancia.capacidad_vehiculo
        return all(carga <= capacidad for carga in self._cargas)
    
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
        
        # MEJORA: Mostrar solo rutas no vacías
        rutas_usadas = [(i, ruta, cargas[i]) for i, ruta in enumerate(self.rutas) if ruta]
        
        for i, ruta, carga in rutas_usadas:
            ruta_str = f"{deposito} -> " + " -> ".join(map(str, ruta)) + f" -> {deposito}"
            lineas.append(f"  Vehículo {i}: {ruta_str} (Carga: {carga}/{self.instancia.capacidad_vehiculo})")
        
        lineas.append(f"  Vehículos utilizados: {len(rutas_usadas)}/{self.instancia.num_vehiculos}")
        
        return "\n".join(lineas)


# ============================
# CLASE PARA REGISTRO DE RESULTADOS
# ============================
class LogResultados:
    """Clase para registrar y generar tabla de resultados de múltiples ejecuciones"""
    
    def __init__(self, nombre_instancia):
        self.nombre_instancia = nombre_instancia
        self.resultados = {}  # Diccionario por instancia
    
    def agregar_ejecucion(self, instancia, numero_ejecucion, costo, tiempo=None):
        """Agrega el resultado de una ejecución"""
        if instancia not in self.resultados:
            self.resultados[instancia] = []
        self.resultados[instancia].append({
            'ejecucion': numero_ejecucion,
            'costo': costo,
            'tiempo': tiempo
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
        
        print(f"[OK] Resultados exportados a: {nombre_archivo}")


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
    
    nombre = ""
    dimension = 0
    capacidad = 0
    coordenadas = {}
    demandas = {}
    
    seccion_actual = None
    
    for linea in lineas:
        linea = linea.strip()
        
        if not linea or linea == 'EOF':
            continue
        
        if linea.startswith('NAME'):
            nombre = linea.split(':')[1].strip()
        elif linea.startswith('DIMENSION'):
            dimension = int(linea.split(':')[1].strip())
        elif linea.startswith('CAPACITY'):
            capacidad = int(linea.split(':')[1].strip())
        elif linea == 'NODE_COORD_SECTION':
            seccion_actual = 'coords'
        elif linea == 'DEMAND_SECTION':
            seccion_actual = 'demands'
        elif linea == 'DEPOT_SECTION':
            seccion_actual = 'depot'
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
    
    instancia = InstanciaCVRP(nombre, nodos, capacidad)
    return instancia


# ============================
# GENERACIÓN DE SOLUCIÓN INICIAL
# ============================
def generar_solucion_inicial_voraz(instancia):
    """
    Genera una solución inicial usando heurística del vecino más cercano
    MEJORADO: Optimización de rendimiento y mejor manejo de capacidad
    """
    clientes_no_asignados = set(c.id for c in instancia.clientes)
    rutas = [[] for _ in range(instancia.num_vehiculos)]
    cargas = [0] * instancia.num_vehiculos
    vehiculo_actual = 0
    
    deposito_id = instancia.deposito.id
    demandas = instancia.demandas_dict  # MEJORA: Usar diccionario pre-calculado
    distancia_func = instancia.distancia  # MEJORA: Cache de función
    
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
                distancia = distancia_func(ultimo_nodo, cliente_id)
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
    
    solucion = Solucion(instancia, rutas)
    
    # Verificar factibilidad
    if not solucion.es_factible():
        print("[ERROR CRÍTICO] No se pudo generar una solución inicial factible")
        cargas_calc = solucion.calcular_cargas()
        for i, carga in enumerate(cargas_calc):
            if carga > instancia.capacidad_vehiculo:
                print(f"  Vehículo {i}: Carga {carga} > Capacidad {instancia.capacidad_vehiculo}")
    
    return solucion


# ============================
# OPERADORES DE VECINDAD
# ============================
def swap_intra_ruta(solucion):
    """Operador: intercambia dos clientes dentro de la misma ruta"""
    nueva_solucion = solucion.copiar()
    rutas_no_vacias = [i for i, r in enumerate(nueva_solucion.rutas) if len(r) >= 2]
    
    if not rutas_no_vacias:
        return nueva_solucion
    
    idx_ruta = random.choice(rutas_no_vacias)
    ruta = nueva_solucion.rutas[idx_ruta]
    i, j = random.sample(range(len(ruta)), 2)
    ruta[i], ruta[j] = ruta[j], ruta[i]
    nueva_solucion.invalidar_cache()
    return nueva_solucion


def swap_inter_ruta(solucion):
    """Operador: intercambia un cliente entre dos rutas distintas"""
    nueva_solucion = solucion.copiar()
    instancia = solucion.instancia
    rutas_no_vacias = [i for i, r in enumerate(nueva_solucion.rutas) if len(r) > 0]
    
    if len(rutas_no_vacias) < 2:
        return nueva_solucion
    
    idx_ruta1, idx_ruta2 = random.sample(rutas_no_vacias, 2)
    pos1 = random.randint(0, len(nueva_solucion.rutas[idx_ruta1]) - 1)
    pos2 = random.randint(0, len(nueva_solucion.rutas[idx_ruta2]) - 1)
    
    cliente1 = nueva_solucion.rutas[idx_ruta1][pos1]
    cliente2 = nueva_solucion.rutas[idx_ruta2][pos2]
    
    demandas = instancia.demandas_dict  # MEJORA: Usar diccionario pre-calculado
    
    # Calcular cargas actuales (MEJORA: más eficiente)
    carga1 = sum(demandas[c] for c in nueva_solucion.rutas[idx_ruta1])
    carga2 = sum(demandas[c] for c in nueva_solucion.rutas[idx_ruta2])
    
    nueva_carga1 = carga1 - demandas[cliente1] + demandas[cliente2]
    nueva_carga2 = carga2 - demandas[cliente2] + demandas[cliente1]
    
    if (nueva_carga1 <= instancia.capacidad_vehiculo and
        nueva_carga2 <= instancia.capacidad_vehiculo):
        nueva_solucion.rutas[idx_ruta1][pos1] = cliente2
        nueva_solucion.rutas[idx_ruta2][pos2] = cliente1
        nueva_solucion.invalidar_cache()
    
    return nueva_solucion


def relocate(solucion):
    """Operador: mueve un cliente de una ruta a otra"""
    nueva_solucion = solucion.copiar()
    instancia = solucion.instancia
    rutas_no_vacias = [i for i, r in enumerate(nueva_solucion.rutas) if len(r) > 0]
    
    if not rutas_no_vacias:
        return nueva_solucion
    
    idx_origen = random.choice(rutas_no_vacias)
    ruta_origen = nueva_solucion.rutas[idx_origen]
    pos = random.randint(0, len(ruta_origen) - 1)
    cliente = ruta_origen[pos]
    
    rutas_disponibles = [i for i in range(instancia.num_vehiculos) if i != idx_origen]
    if not rutas_disponibles:
        return nueva_solucion
    
    idx_destino = random.choice(rutas_disponibles)
    demandas = instancia.demandas_dict  # MEJORA
    carga_destino = sum(demandas[c] for c in nueva_solucion.rutas[idx_destino])
    
    if carga_destino + demandas[cliente] <= instancia.capacidad_vehiculo:
        ruta_origen.pop(pos)
        pos_insercion = random.randint(0, len(nueva_solucion.rutas[idx_destino])) if nueva_solucion.rutas[idx_destino] else 0
        nueva_solucion.rutas[idx_destino].insert(pos_insercion, cliente)
        nueva_solucion.invalidar_cache()
    
    return nueva_solucion


def reverse_subruta(solucion):
    """Operador 2-opt: invierte un segmento de una ruta"""
    nueva_solucion = solucion.copiar()
    rutas_validas = [i for i, r in enumerate(nueva_solucion.rutas) if len(r) >= 2]
    
    if not rutas_validas:
        return nueva_solucion
    
    idx_ruta = random.choice(rutas_validas)
    ruta = nueva_solucion.rutas[idx_ruta]
    i, j = sorted(random.sample(range(len(ruta) + 1), 2))
    ruta[i:j] = reversed(ruta[i:j])
    nueva_solucion.invalidar_cache()
    return nueva_solucion


def two_opt_inter_route(solucion):
    """
    Operador 2-opt inter-ruta VERDADERO:
    Selecciona una arista en cada ruta, las rompe y reconecta de forma cruzada.
    Esto puede eliminar cruces entre rutas y mejorar la solución global.
    """
    nueva_solucion = solucion.copiar()
    instancia = solucion.instancia
    
    if instancia.num_vehiculos < 2:
        return nueva_solucion
    
    # Seleccionar dos rutas distintas no vacías
    rutas_no_vacias = [i for i, r in enumerate(nueva_solucion.rutas) if len(r) >= 2]
    
    if len(rutas_no_vacias) < 2:
        return nueva_solucion
    
    r1, r2 = random.sample(rutas_no_vacias, 2)
    ruta1 = nueva_solucion.rutas[r1][:]
    ruta2 = nueva_solucion.rutas[r2][:]
    
    # Seleccionar posiciones de corte (aristas)
    # Posición i significa cortar entre ruta1[i] y ruta1[i+1]
    i = random.randint(0, len(ruta1) - 1)
    j = random.randint(0, len(ruta2) - 1)
    
    # Crear nuevas rutas aplicando 2-opt inter-ruta
    # Ruta1: mantiene [0..i] + toma [j+1..fin] de ruta2 (invertido)
    # Ruta2: mantiene [0..j] + toma [i+1..fin] de ruta1 (invertido)
    nueva1 = ruta1[:i+1] + ruta2[j+1:][::-1]
    nueva2 = ruta2[:j+1] + ruta1[i+1:][::-1]
    
    # Validar capacidad
    demandas = instancia.demandas_dict
    demanda1 = sum(demandas[c] for c in nueva1)
    demanda2 = sum(demandas[c] for c in nueva2)
    
    if demanda1 <= instancia.capacidad_vehiculo and demanda2 <= instancia.capacidad_vehiculo:
        nueva_solucion.rutas[r1] = nueva1
        nueva_solucion.rutas[r2] = nueva2
        nueva_solucion.invalidar_cache()
    
    return nueva_solucion


def exchange_inter_route(solucion):
    """
    Operador Exchange/Segment Swap inter-ruta:
    Intercambia segmentos completos entre dos rutas sin invertir.
    Este es el operador que estaba implementado antes como "two_opt_inter_route".
    """
    nueva_solucion = solucion.copiar()
    instancia = solucion.instancia
    
    if instancia.num_vehiculos < 2:
        return nueva_solucion
    
    rutas_no_vacias = [i for i, r in enumerate(nueva_solucion.rutas) if len(r) >= 2]
    
    if len(rutas_no_vacias) < 2:
        return nueva_solucion
    
    r1, r2 = random.sample(rutas_no_vacias, 2)
    ruta1 = nueva_solucion.rutas[r1]
    ruta2 = nueva_solucion.rutas[r2]
    
    # Cortes aleatorios
    i = random.randint(1, len(ruta1) - 1)
    j = random.randint(1, len(ruta2) - 1)
    
    # Intercambiar segmentos (sin invertir)
    nueva1 = ruta1[:i] + ruta2[j:]
    nueva2 = ruta2[:j] + ruta1[i:]
    
    # Validar capacidad
    demandas = instancia.demandas_dict
    demanda1 = sum(demandas[c] for c in nueva1)
    demanda2 = sum(demandas[c] for c in nueva2)
    
    if demanda1 <= instancia.capacidad_vehiculo and demanda2 <= instancia.capacidad_vehiculo:
        nueva_solucion.rutas[r1] = nueva1
        nueva_solucion.rutas[r2] = nueva2
        nueva_solucion.invalidar_cache()
    
    return nueva_solucion


def generar_vecino(solucion):
    """Genera una solución vecina aplicando un operador aleatorio"""
    operadores = [
        swap_intra_ruta,
        swap_inter_ruta,
        relocate,
        reverse_subruta,
        two_opt_inter_route
    ]
    
    operador = random.choice(operadores)
    return operador(solucion)


# ============================
# RECOCIDO SIMULADO
# ============================
def criterio_metropolis(costo_actual, costo_nuevo, temperatura):
    """Criterio de aceptación de Metropolis"""
    if costo_nuevo < costo_actual:
        return True
    
    if temperatura <= 0:
        return False
    
    delta = costo_nuevo - costo_actual
    probabilidad = math.exp(-delta / temperatura)
    return random.random() < probabilidad


def recocido_simulado(instancia, solucion_inicial=None, verbose=True):
    """Algoritmo de Recocido Simulado para CVRP"""
    inicio = time.time()
    
    if solucion_inicial is None:
        if verbose:
            print("Generando solución inicial voraz...")
        solucion_inicial = generar_solucion_inicial_voraz(instancia)
    
    if not solucion_inicial.es_factible():
        print("[ERROR CRÍTICO] La solución inicial no es factible.")
        return None, None
    
    solucion_actual = solucion_inicial.copiar()
    costo_actual = solucion_actual.get_costo()
    
    mejor_solucion = solucion_actual.copiar()
    mejor_costo = costo_actual
    
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
    
    # MEJORA: Contador de intentos fallidos para debugging
    intentos_fallidos = 0
    total_intentos = 0
    
    while temperatura > TEMPERATURA_MINIMA and iteraciones_sin_mejora < MAX_ITERACIONES_SIN_MEJORA:
        for _ in range(ITERACIONES_POR_TEMPERATURA):
            iteracion += 1
            iteraciones_sin_mejora += 1
            total_intentos += 1
            
            solucion_vecina = generar_vecino(solucion_actual)
            
            if not solucion_vecina.es_factible():
                intentos_fallidos += 1
                continue
            
            costo_vecino = solucion_vecina.get_costo()
            
            if criterio_metropolis(costo_actual, costo_vecino, temperatura):
                solucion_actual = solucion_vecina
                costo_actual = costo_vecino
                
                if costo_actual < mejor_costo:
                    mejor_solucion = solucion_actual.copiar()
                    mejor_costo = costo_actual
                    iteraciones_sin_mejora = 0
                    
                    if verbose:
                        print(f"Iter {iteracion:6d} | T={temperatura:8.2f} | Mejor costo: {mejor_costo:.2f} ✓")
        
        historial_mejor_costo.append(mejor_costo)
        historial_temperatura.append(temperatura)
        historial_iteracion.append(iteracion)
        
        temperatura *= ALPHA
    
    tiempo_total = time.time() - inicio
    
    if not mejor_solucion.es_factible():
        print("[ERROR] La mejor solución encontrada no es factible.")
    
    if verbose:
        print(f"\n=== RESULTADO FINAL ===")
        print(f"Costo inicial: {solucion_inicial.get_costo():.2f}")
        print(f"Mejor costo encontrado: {mejor_costo:.2f}")
        mejora = ((solucion_inicial.get_costo() - mejor_costo) / solucion_inicial.get_costo()) * 100
        print(f"Mejora: {mejora:.2f}%")
        print(f"Iteraciones totales: {iteracion}")
        print(f"Tiempo: {tiempo_total:.2f} segundos")
        # MEJORA: Estadísticas de intentos
        tasa_exito = ((total_intentos - intentos_fallidos) / total_intentos * 100) if total_intentos > 0 else 0
        print(f"Tasa de éxito de vecinos factibles: {tasa_exito:.1f}%")
        print(f"\n{mejor_solucion}")
    
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
    
    # MEJORA: Usar diferentes estilos de línea si hay muchos vehículos
    colores = plt.cm.tab20(range(len(solucion.rutas)))
    estilos_linea = ['-', '--', '-.', ':']
    
    rutas_dibujadas = 0
    for i, ruta in enumerate(solucion.rutas):
        if not ruta:
            continue
        
        ruta_completa = [deposito] + ruta + [deposito]
        xs = [coords[nodo][0] for nodo in ruta_completa]
        ys = [coords[nodo][1] for nodo in ruta_completa]
        
        # MEJORA: Alternar estilos de línea para mejor diferenciación
        estilo = estilos_linea[i % len(estilos_linea)] if len(solucion.rutas) > 20 else '-'
        
        ax.plot(xs, ys, marker='o', linestyle=estilo, color=colores[i], 
                linewidth=2, markersize=6, label=f'Vehículo {i}')
        rutas_dibujadas += 1
    
    # Marcar depósito
    ax.plot(coords[deposito][0], coords[deposito][1], 's',
            color='red', markersize=15, label='Depósito', zorder=5)
    
    # MEJORA: Leyenda más compacta
    if rutas_dibujadas > 15:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                  ncol=2, fontsize=7, framealpha=0.9)
    else:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                  fontsize=9, framealpha=0.9)
    
    ax.set_title(f"{titulo}\nCosto total: {solucion.get_costo():.2f}", fontsize=12, fontweight='bold')
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
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Gráfico de mejor costo
    ax1.plot(historial['iteraciones'], historial['mejor_costo'],
             'b-', linewidth=2, label='Mejor costo')
    ax1.set_xlabel('Iteración', fontsize=10)
    ax1.set_ylabel('Mejor Costo', fontsize=10)
    ax1.set_title('Evolución del Mejor Costo', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Gráfico de temperatura
    ax2.plot(historial['iteraciones'], historial['temperatura'],
             'r-', linewidth=2, label='Temperatura')
    ax2.set_xlabel('Iteración', fontsize=10)
    ax2.set_ylabel('Temperatura (escala log)', fontsize=10)
    ax2.set_title('Esquema de Enfriamiento', fontsize=11, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
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
    print("     Versión Mejorada con Estadísticas")
    print("="*60)
    
    archivos_instancias = [
        "Facil.vrp",
        "Medio.vrp",
        "Dificil.vrp"
    ]
    
    log_global = LogResultados("Todas las Instancias")
    resultados_por_instancia = {}
    
    for i, archivo in enumerate(archivos_instancias, 1):
        print(f"\n{'='*60}")
        print(f"  INSTANCIA {i}: {archivo}")
        print(f"{'='*60}")
        
        try:
            # Leer instancia
            print(f"Cargando instancia {archivo}...")
            instancia = leer_archivo_vrp(archivo)
            print(f"[OK] Instancia cargada: {len(instancia.clientes)} clientes, "
                  f"{instancia.num_vehiculos} vehículos, capacidad {instancia.capacidad_vehiculo}")
            
            # Ejecutar múltiples veces
            print(f"\nEjecutando {NUMERO_EJECUCIONES} ejecuciones...")
            
            nombre_corto = archivo.replace('.vrp', '')
            resultados_por_instancia[nombre_corto] = []
            
            # MEJORA: Barra de progreso simple
            for ejecucion in range(1, NUMERO_EJECUCIONES + 1):
                print(f"\n--- Ejecución {ejecucion}/{NUMERO_EJECUCIONES} ---")
                
                # Cambiar semilla para cada ejecución
                random.seed(SEMILLA_ALEATORIA + ejecucion)
                
                # Ejecutar recocido simulado
                inicio_ejec = time.time()
                mejor_solucion, historial = recocido_simulado(instancia, verbose=False)
                tiempo_ejec = time.time() - inicio_ejec
                
                if mejor_solucion is None:
                    print(f"[ERROR] No se pudo resolver la ejecución {ejecucion}")
                    continue
                
                costo_final = mejor_solucion.get_costo()
                print(f"Costo obtenido: {costo_final:.2f} | Tiempo: {tiempo_ejec:.2f}s")
                
                # Registrar resultado
                log_global.agregar_ejecucion(nombre_corto, ejecucion, costo_final, tiempo_ejec)
                resultados_por_instancia[nombre_corto].append({
                    'solucion': mejor_solucion,
                    'historial': historial,
                    'costo': costo_final,
                    'tiempo': tiempo_ejec
                })
            
            # Mostrar mejor solución de esta instancia
            if resultados_por_instancia[nombre_corto]:
                mejor_idx = min(range(len(resultados_por_instancia[nombre_corto])),
                               key=lambda x: resultados_por_instancia[nombre_corto][x]['costo'])
                
                mejor_resultado = resultados_por_instancia[nombre_corto][mejor_idx]
                
                print(f"\n{'='*60}")
                print(f"MEJOR SOLUCIÓN DE {nombre_corto.upper()}")
                print(f"{'='*60}")
                print(f"Ejecución: {mejor_idx + 1}")
                print(f"Tiempo: {mejor_resultado['tiempo']:.2f}s")
                print(mejor_resultado['solucion'])
                
                # MEJORA: Mostrar estadísticas de todas las ejecuciones
                costos = [r['costo'] for r in resultados_por_instancia[nombre_corto]]
                tiempos = [r['tiempo'] for r in resultados_por_instancia[nombre_corto]]
                print(f"\nEstadísticas de {NUMERO_EJECUCIONES} ejecuciones:")
                print(f"  Mejor costo: {min(costos):.2f}")
                print(f"  Peor costo: {max(costos):.2f}")
                print(f"  Costo medio: {np.mean(costos):.2f} ± {np.std(costos, ddof=1):.2f}")
                print(f"  Tiempo medio: {np.mean(tiempos):.2f}s ± {np.std(tiempos, ddof=1):.2f}s")
                
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
    if resultados_por_instancia:
        print("\n" + "="*60)
        print("     GENERANDO TABLA DE RESULTADOS")
        print("="*60)
        
        log_global.generar_tabla()
        
        # Exportar a CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_csv = f"resultados_cvrp_{timestamp}.csv"
        log_global.exportar_csv(nombre_csv)
        
        # ============================
        # RESUMEN FINAL DETALLADO
        # ============================
        print(f"\n{'='*60}")
        print("     RESUMEN FINAL DE RESULTADOS")
        print(f"{'='*60}\n")
        
        for nombre_inst, resultados in sorted(resultados_por_instancia.items()):
            if resultados:
                costos = [r['costo'] for r in resultados]
                tiempos = [r['tiempo'] for r in resultados]
                
                print(f"{nombre_inst.upper()}:")
                print(f"  Mejor costo:       {min(costos):12.2f}")
                print(f"  Peor costo:        {max(costos):12.2f}")
                print(f"  Costo medio:       {np.mean(costos):12.2f}")
                print(f"  Desv. estándar:    {np.std(costos, ddof=1):12.2f}")
                print(f"  Coef. variación:   {(np.std(costos, ddof=1) / np.mean(costos) * 100):11.2f}%")
                print(f"  Tiempo medio:      {np.mean(tiempos):12.2f}s")
                print()
        
        print("="*60)
        print("Proceso completado exitosamente.")
        print(f"Resultados guardados en: {nombre_csv}")
        print("="*60)


if __name__ == "__main__":
    main()