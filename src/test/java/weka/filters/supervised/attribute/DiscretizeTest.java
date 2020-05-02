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

package weka.filters.supervised.attribute;

import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.AbstractFilterTest;
import weka.filters.Filter;

import junit.framework.Test;
import junit.framework.TestSuite;

/**
 * Tests Discretize. Run from the command line with:<p>
 * java weka.filters.supervised.attribute.DiscretizeTest
 *
 * @author <a href="mailto:len@reeltwo.com">Len Trigg</a>
 * @version $Revision: 1.3 $
 */
public class DiscretizeTest extends AbstractFilterTest {
  
  public DiscretizeTest(String name) { super(name);  }

  /** Need to set the class index */
  protected void setUp() throws Exception {

    super.setUp();
    m_Instances.setClassIndex(1);
  }

  /** Creates a default Discretize */
  public Filter getFilter() {
    Discretize f= new Discretize();
    return f;
  }

  /** Creates a specialized Discretize */
  public Filter getFilter(String rangelist) {
    
    try {
      Discretize f = new Discretize();
      f.setAttributeIndices(rangelist);
      return f;
    } catch (Exception ex) {
      ex.printStackTrace();
      fail("Exception setting attribute range: " + rangelist 
           + "\n" + ex.getMessage()); 
    }
    return null;
  }

  public void testTypical() {
    m_Filter = getFilter("1,2");
    Instances result = useFilter();
    assertEquals(m_Instances.numAttributes(), result.numAttributes());
    // None of the attributes should have changed, since 1,2 aren't numeric
    for (int i = 0; i < result.numAttributes(); i++) {
      assertEquals(m_Instances.attribute(i).type(), result.attribute(i).type());
      assertEquals(m_Instances.attribute(i).name(), result.attribute(i).name());
    }
  }

  public void testTypical2() {
    m_Filter = getFilter("3-4");
    Instances result = useFilter();
    assertEquals(m_Instances.numAttributes(), result.numAttributes());
    for (int i = 0; i < result.numAttributes(); i++) {
      if (i != 2) {
        assertEquals(m_Instances.attribute(i).type(), result.attribute(i).type());
        assertEquals(m_Instances.attribute(i).name(), result.attribute(i).name());
      } else {
        assertEquals(Attribute.NOMINAL, result.attribute(i).type());
        assertEquals(1, result.attribute(i).numValues());
      }
    }
  }

  public void testInverted() {
    m_Filter = getFilter("1,2");
    ((Discretize)m_Filter).setInvertSelection(true);
    Instances result = useFilter();
    assertEquals(m_Instances.numAttributes(), result.numAttributes());
    for (int i = 0; i < result.numAttributes(); i++) {
      if ((i < 2) || !m_Instances.attribute(i).isNumeric()) {
        assertEquals(m_Instances.attribute(i).type(), result.attribute(i).type());
        assertEquals(m_Instances.attribute(i).name(), result.attribute(i).name());
      } else {
        assertEquals(Attribute.NOMINAL, result.attribute(i).type());
        assertEquals(1, result.attribute(i).numValues());
      }
    }
  }

  public void testNonInverted2() {
    m_Filter = getFilter("first-3");
    ((Discretize)m_Filter).setInvertSelection(true);
    Instances result = useFilter();
    assertEquals(m_Instances.numAttributes(), result.numAttributes());
    for (int i = 0; i < result.numAttributes(); i++) {
      if ((i < 3) || !m_Instances.attribute(i).isNumeric()) {
        assertEquals(m_Instances.attribute(i).type(), result.attribute(i).type());
        assertEquals(m_Instances.attribute(i).name(), result.attribute(i).name());
      } else {
        assertEquals(Attribute.NOMINAL, result.attribute(i).type());
        assertEquals(1, result.attribute(i).numValues());
      }
    }
  }

  public void testBetterEncoding() {
    m_Filter = getFilter("3");
    ((Discretize)m_Filter).setUseBetterEncoding(true);
    Instances result = useFilter();
    assertEquals(m_Instances.numAttributes(), result.numAttributes());
    assertEquals(Attribute.NOMINAL, result.attribute(2).type());
  }

  public void testUseKononenko() {
    m_Filter = getFilter("3");
    ((Discretize)m_Filter).setUseKononenko(true);
    Instances result = useFilter();
    assertEquals(m_Instances.numAttributes(), result.numAttributes());
    assertEquals(Attribute.NOMINAL, result.attribute(2).type());
  }

  public static Test suite() {
    return new TestSuite(DiscretizeTest.class);
  }

  public static void main(String[] args){
    junit.textui.TestRunner.run(suite());
  }

}
