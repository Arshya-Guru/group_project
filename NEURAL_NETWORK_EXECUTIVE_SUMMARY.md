# Neural Network Performance - Executive Summary

## TL;DR: Model Failed Spectacularly

**Performance**: 0.6299 AUC (vs XGBoost's 0.924 AUC)
**Gap**: -31.8% worse than XGBoost
**Clinical Utility**: ❌ UNSUITABLE FOR DEPLOYMENT
**Root Cause**: Over-regularization + misconfigured loss function

---

## The Numbers

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Test AUC | >0.90 | **0.6299** | ❌ 30% below |
| Test Accuracy | >85% | **23.46%** | ❌ Worse than random |
| CV Stability (std) | <0.05 | **0.1580** | ❌ 3× too variable |
| Log Loss | <0.50 | **1.2856** | ❌ Poorly calibrated |

---

## What Went Wrong

### 1. Excessive Regularization = Underfitting
- **Dropout 0.5** (50%) on only **12 hidden units** = 6 active neurons
- 6 neurons cannot learn complex medical patterns
- Model crippled before it could learn

### 2. Focal Loss Configured Backwards
```python
FocalLoss(alpha=0.25, gamma=2.0)  # WRONG
# alpha=0.25 DOWNWEIGHTS minority class (opposite of intended!)
```
Should be: `alpha=0.75` to emphasize dementia cases

### 3. Architecture Too Tiny
- 277 trainable parameters for 324 samples
- With 50% dropout = ~138 active parameters
- **1.08 samples per parameter** (need 10+)
- Insufficient capacity to learn patterns

### 4. Never Converged
- All models hit 200 epoch limit
- Early stopping never triggered
- Learning rate too low (0.001)
- Models needed more training, not less

---

## Most Damning Evidence

### Test Accuracy: 23.46%

The model predicts **everything as dementia (CDR>0)**:
```
Predicted all 81 test samples as CDR>0
Correct: 19 actual dementia cases = 23.46% accuracy
Missed: All 62 healthy patients
```

This is **worse than predicting all healthy** (77% accuracy).

### Cross-Validation Fold 5: 0.2643 AUC

One fold performed **worse than random guessing** (0.50 AUC):
- Gap below random: -0.236 AUC
- This fold exposed complete failure to generalize

---

## Quick Fixes (Could Improve to ~0.90 AUC)

### Immediate Changes

```python
# 1. Fix focal loss alpha
FocalLoss(alpha=0.75, gamma=2.0)  # Correct minority class focus

# 2. Reduce dropout
Dropout(0.3)  # Down from 0.5

# 3. Increase learning rate
Adam(learning_rate=0.01)  # Up from 0.001

# 4. Larger architecture
Dense(64) → Dense(32) → Dense(1)  # Up from Dense(12) → Dense(1)

# 5. Train longer
epochs=500, patience=50  # Up from 200 epochs
```

**Expected Impact**: +0.25 to +0.27 AUC → Final: 0.88-0.90 AUC

---

## Why XGBoost Won (0.924 vs 0.630 AUC)

| Factor | XGBoost | Neural Network |
|--------|---------|----------------|
| **Tabular data handling** | Native | Requires careful design |
| **Small data robustness** | Excellent | Poor |
| **Training time** | 1 minute | 10 minutes |
| **Hyperparameter sensitivity** | Low | Very high |
| **Out-of-box performance** | 0.924 AUC | 0.630 AUC |
| **Stability (CV std)** | 0.020 | 0.158 |

**Verdict**: XGBoost is **fundamentally better suited** for small tabular medical data

---

## Recommendations

### Short Term: Do NOT Deploy Neural Network
- Current model is **clinically dangerous**
- 77% misdiagnosis rate
- No better than predicting everyone has dementia

### Medium Term: Fix Configuration & Re-evaluate
- Implement 5 quick fixes above
- Re-run full nested CV
- Target: 0.88-0.90 AUC (competitive with XGBoost)
- Timeline: 1-2 days of work

### Long Term: Use XGBoost for Production
- **More reliable**: 0.924 AUC with 0.020 std
- **Faster**: 5× faster training and inference
- **Simpler**: Less hyperparameter tuning
- **Better interpretability**: SHAP values

### When to Use Neural Networks
Neural networks would excel if we had:
- **100× more data** (30,000+ samples)
- **Raw MRI images** (3D CNNs)
- **Longitudinal sequences** (RNNs/LSTMs)
- **Multimodal data** (images + tabular + clinical notes)

For current task: **Stick with XGBoost**

---

## Key Lessons

1. **Over-regularization is real** - You can regularize too aggressively
2. **Hyperparameters matter enormously** - Wrong focal loss alpha killed performance
3. **Convergence > Epochs** - Models stopped before learning anything useful
4. **Right tool for the job** - Tree methods beat NNs on small tabular data
5. **Ensemble can't fix bad base models** - Need good individuals first

---

## Comparison to Expectations

**Design Doc Predicted**: 0.88-0.91 AUC (likely case)
**Actual Result**: 0.63 AUC
**Miss**: -28% to -30%

Performance was **below the worst-case scenario** (0.85 AUC).

---

## Bottom Line

### What We Built
✅ Methodologically rigorous implementation
✅ Proper nested CV, no data leakage
✅ Well-documented, reproducible code

### What We Got
❌ Model that performs worse than baseline
❌ 31.8% below XGBoost
❌ Unsuitable for clinical use

### What We Learned
💡 Configuration matters more than architecture
💡 XGBoost is the right tool for small tabular data
💡 Neural networks need careful tuning on limited samples

---

**Recommendation**: Use XGBoost (0.924 AUC) for production. If neural network is required, implement all 5 fixes and re-evaluate.

**Status**: Model REJECTED for deployment
**Next Steps**: Fix configuration or abandon neural network approach

---

*Full analysis: See `neural_network_performance_scrutiny.md`*
*Notebook: See `neural_network_analysis.ipynb`*
*Design rationale: See `neural_network_design_decisions.md`*
