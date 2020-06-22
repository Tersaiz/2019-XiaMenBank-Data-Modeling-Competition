//feature_selector主要对以下类型的特征进行选择：

具有高missing-values百分比的特征 
具有高相关性的特征
对模型预测结果无贡献的特征（即zero importance）
对模型预测结果只有很小贡献的特征（即low importance）
具有单个值的特征（即数据集中该特征取值的集合只有一个元素）

针对上面五种类型的特征，feature-selector分别提供以下五个函数来对此处理：

identify_missing(*)       输入参数missing_threshold，主要用来控制缺失数据占比，当缺失数据量大于missing_threshold时，将其筛选
identify_collinear(*)      输入参数correlation_threshold，用来控制两个特征间相关系数，one_hot=False可选
identify_zero_importance(*)     输入参数主要是task和eval_metric两个参数，分别选择任务类型和验证指标
identify_low_importance(*)       输入参数cumulative_importance，用来控制当特征重要性到达cumulative_importance时，剩下的特征均是低重要性特征
identify_single_unique(*)     无需输入参数


identify_missing(*)的设计思路是计算得到缺失值百分比，与missing_threshold相比较，当缺失值百分百大于missing_threshold时，将其加入到筛选集合中
identify_single_unique的设计思路是先得到每个特征唯一值的数量，取出唯一值数量为1的特征，将其加入到筛选集合中
identify_collinear(*) 的设计思路是：
1）根据参数'one_hot'对数据集特征进行one-hot encoding（调用pd.get_dummies方法）。如果'one_hot=True'则对特征将进行one-hot encoding，
并将编码的特征与原数据集整合起来组成新的数据集，如果'one_hot=False'则什么不做，进入下一步；
2) 计算步骤1得出数据集的相关矩阵 [公式] (通过DataFrame.corr()，注意 [公式] 也为一个DateFrame)，并取相关矩阵的上三角部分得到 [公式] ；
3) 遍历 [公式] 的每一列(即每一个特征)，如果该列的任何一个相关值大于correlation_threshold，则取出该列，并放到一个列表中
（该列表中的feature，即具有high 相关性的特征，之后会从数据集去除）
在设计过程中有一个问题，就是他将高相关性的特征全部删除了，并没有取出相关特征中，不相关的特征加入到网络中，因而这里不建议使用

identify_zero_importance(*)和identify_low_importance(*)的思路都是通过lightgbm来得到特征重要性然后再将特征重要性总和化成1，然后将特征重要性排序，将排序后的特征
做cumsum（），a=[1,2,3]->cumsum(a)->[1,3,6],得到累加的特征重要性，然后对大于cumulative_importance后，剩下的特征均为低重要性特征

identify_all(self, selection_params) 是对上述五种函数同时应用

remove(self, methods, keep_one_hot=True):
methods值得主要是选择使用那几个函数

*/