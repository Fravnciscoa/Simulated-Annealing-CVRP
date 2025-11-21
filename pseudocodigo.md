ALGORITMO: RecocidoSimulado(instancia)

  // Inicialización
  solución ← GenerarSoluciónVoraz(instancia)
  mejor_solución ← solución
  temperatura ← T_inicial
  iteraciones_sin_mejora ← 0
  
  // Bucle principal
  MIENTRAS temperatura > T_mínima Y iteraciones_sin_mejora < MAX_iter_sin_mejora:
    
    PARA i = 1 HASTA iteraciones_por_temperatura:
      vecino ← GenerarVecino(solución)
      
      SI EsFactible(vecino):
        Δ ← Costo(vecino) - Costo(solución)
        
        // Criterio Metropolis
        SI Δ < 0 O random() < exp(-Δ/temperatura):
          solución ← vecino
          
          SI Costo(solución) < Costo(mejor_solución):
            mejor_solución ← solución
            iteraciones_sin_mejora ← 0
    
    temperatura ← temperatura × α  // Enfriamiento geométrico
  
  RETORNAR mejor_solución