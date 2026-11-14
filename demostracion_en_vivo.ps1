# Script de Demostración en Vivo para la Defensa del Proyecto PLN
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "     SISTEMA DE RECUPERACIÓN Y APRENDIZAJE AUTOMÁTICO (PLN)     " -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "1. Abrir Dashboard Visual de Presentación (HTML)" -ForegroundColor White
Write-Host "2. Demostración en vivo de Recuperación de Información (Fase 1)" -ForegroundColor White
Write-Host "3. Ver métricas de Clasificación Supervisada (Fase 2)" -ForegroundColor White
Write-Host "4. Ver interpretación de Clústeres K-Means (Fase 2)" -ForegroundColor White
Write-Host "5. Abrir Informe Académico Completo (Markdown)" -ForegroundColor White
Write-Host "6. Ejecutar evaluación rápida de clasificación (evaluate_final.py)" -ForegroundColor White
Write-Host "7. Salir" -ForegroundColor Gray
Write-Host "-----------------------------------------------------------------" -ForegroundColor Cyan

$opcion = Read-Host "Selecciona una opción [1-7]"

switch ($opcion) {
    "1" {
        Start-Process "docs\presentacion_proyecto_pln.html"
    }
    "2" {
        $consulta = Read-Host "Escribe una consulta de búsqueda (ejemplo: 'climate change and renewable energy')"
        if ([string]::IsNullOrWhiteSpace($consulta)) {
            $consulta = "artificial intelligence regulation"
        }
        python fase_1_recuperacion\benchmark\main.py search --query $consulta
    }
    "3" {
        Get-Content "fase_2_aprendizaje\clasificacion\resultados\classification_metrics.tsv"
    }
    "4" {
        Get-Content "fase_2_aprendizaje\agrupamiento\resultados\cluster_interpretation.tsv"
    }
    "5" {
        Start-Process "docs\INFORME_ETAPA_2.md"
    }
    "6" {
        python "fase_2_aprendizaje\clasificacion\evaluate_final.py"
    }
    default {
        Write-Host "Saliendo..." -ForegroundColor Gray
    }
}
