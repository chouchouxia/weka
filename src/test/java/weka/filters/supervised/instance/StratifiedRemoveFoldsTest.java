/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * Copyright (C) 2002 University of Waikato 
 */

package weka.filters.supervised.instance;

import weka.core.Instances;
import weka.filters.AbstractFilterTest;
import weka.filters.Filter;

import junit.framework.Test;
import junit.framework.TestSuite;

/**
 * Tests StratifiedRemoveFolds. Run from the command line with:<p>
 * java weka.filters.supervised.instance.StratifiedRemoveFoldsTest
 *
 * @author <a href="mailto:len@reeltwo.com">Len Trigg</a>
 * @version $Revision: 1.3 $
 */
public class StratifiedRemoveFoldsTest extends AbstractFilterTest {
  
  public StratifiedRemoveFoldsTest(String name) { super(name);  }

  /** Creates a default StratifiedRemoveFolds */
  public Filter getFilter() {
    StratifiedRemoveFolds f = new StratifiedRemoveFolds();
    return f;
  }

  /** Remove string attributes from default fixture instances */
  protected void setUp() throws Exception {

    super.setUp();
    m_Instances.setClassIndex(1);
  }

  public void testAllFolds() {
    
    int totInstances = 0;
    for (int i = 0; i < 10; i++) {
      ((StratifiedRemoveFolds)m_Filter).setFold(i + 1);
      Instances result = useFilter();
      assertEquals(m_Instances.numAttributes(), result.numAttributes());
      totInstances += result.numInstances();
    }
    assertEquals("Expecting output number of instances to match",
                 m_Instances.numInstances(),  totInstances);
  }

  public static Test suite() {
    return new TestSuite(StratifiedRemoveFoldsTest.class);
  }

  public static void main(String[] args){
    junit.textui.TestRunner.run(suite());
  }

}
