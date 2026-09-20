# 安装依赖（首次运行执行一次）
# pip install statsmodels islp pandas

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from ISLP import load_data

# 1. 加载数据集
Carseats = load_data('Carseats')

# 2. 构建多元线性回归模型
model = smf.ols(
    formula="Sales ~ Price + Income + Advertising + ShelveLoc",
    data=Carseats
)
result = model.fit()

# 3. 输出回归拟合报告
print("="*60)
print("回归模型拟合报告")
print("="*60)
print(result.summary())

# 4. ShelveLoc相关问题回答
print("\n" + "="*60)
print("ShelveLoc 分析")
print("="*60)
print("ShelveLoc基准组：Bad（货架位置差）")
print("ShelveLoc[Good]系数含义：")
print("在Price、Income、Advertising保持不变的前提下，货架位置Good的门店，销售额平均比货架位置Bad的门店高出该系数数值。")

# 5. 计算VIF（方差膨胀因子，检验多重共线性）
X = result.model.data.exog
var_names = result.model.data.param_names

vif_list = []
for i in range(X.shape[1]):
    vif = variance_inflation_factor(X, i)
    vif_list.append(vif)

vif_df = pd.DataFrame({
    "变量": var_names,
    "VIF值": vif_list
})

print("\n" + "="*60)
print("VIF 多重共线性检验结果")
print("="*60)
print(vif_df)
print("\nVIF判断标准：VIF>5 存在多重共线性风险；VIF>10 严重多重共线性")
