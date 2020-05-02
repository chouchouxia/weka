package weka.classifiers.myalgorithm;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Arrays;

public class MyLinearRegression extends Classifier
{

  public int m_NumInstance;
  public int m_NumAttribute;
  /**
   * 参数列表
   */
  public double m_Parameter[];


  @Override
  public void buildClassifier(Instances data) throws Exception {
    // TODO Auto-generated method stub
    m_NumInstance = data.numInstances();
    m_NumAttribute = data.numAttributes();
    m_Parameter = new double[m_NumAttribute];

    //训练数据集的属性值矩阵
    double [][]mat_X = new double[m_NumInstance][m_NumAttribute];
    //训练数据集的类属性值矩阵
    double [][]mat_Y = new double[m_NumInstance][1];
    for (int i = 0; i < m_NumInstance; i++)
    {
      Instance instance = data.instance(i);
      mat_Y[i][0] = instance.classValue();
      for (int j = 0; j < m_NumAttribute; j++)
      {
        if (j != data.classIndex())
          mat_X[i][j] = instance.value(j);
        else
          mat_X[i][j] = 1.0;
      }
    }

    //最小二乘法求线性回归的参数
    weka.core.matrix.Matrix X = new weka.core.matrix.Matrix(mat_X);
    weka.core.matrix.Matrix Y = new weka.core.matrix.Matrix(mat_Y);
    weka.core.matrix.Matrix tranposeX = X.transpose();
    weka.core.matrix.Matrix temp1 = tranposeX.times(X);
    weka.core.matrix.Matrix temp2 = temp1.inverse();
    weka.core.matrix.Matrix temp3 = temp2.times(tranposeX);
    weka.core.matrix.Matrix temp4 = temp3.times(Y);
    m_Parameter = temp4.getRowPackedCopy();
  }

  @Override
  public double classifyInstance(Instance instance) throws Exception {
    double result = 0.0;

    for (int i = 0; i < m_NumAttribute; i++)
    {
      if (i != instance.classIndex())
        result += m_Parameter[i] * instance.value(i);
      else
        result += m_Parameter[i];
    }

    return result;
  }

  @Override
  public String toString() {
    String res = "m_Parameter=" + "\n\n";
    for (Double tmp : m_Parameter){
      res += tmp.toString().substring(0,7);
      res += "\n\n";
    }
    return res;
  }
}