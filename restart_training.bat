@echo off
REM PTD Training Restart Script
REM Archives buggy checkpoints and starts fresh Phase 3 training

echo ============================================================
echo PTD Training Restart - Bug Fix Applied
echo ============================================================
echo.

REM Create archive directory for buggy checkpoints
if not exist "checkpoints\archived_buggy" mkdir "checkpoints\archived_buggy"

echo Moving buggy Phase 3 checkpoints to archive...
if exist "checkpoints\ptd_phase3_*.pt" (
    move checkpoints\ptd_phase3_*.pt checkpoints\archived_buggy\ >nul 2>&1
    echo   Archived Phase 3 checkpoints
) else (
    echo   No Phase 3 checkpoints to archive
)

echo.
echo ============================================================
echo Starting Phase 3 Training (Fresh)
echo ============================================================
echo.
echo Using Phase 2 router: checkpoints/ptd_student_step003000.pt
echo.
echo Press Ctrl+C to cancel, or
pause

REM Start Phase 3 training
python train_phase3.py --router-ckpt checkpoints/ptd_student_step003000.pt --batch 2

echo.
echo ============================================================
echo Training Complete!
echo ============================================================
echo.
echo Verify results with:
echo   python verify_accuracy.py --sparsity 0.3 --checkpoint checkpoints/ptd_phase3_stage5_keep30.pt
echo.
pause