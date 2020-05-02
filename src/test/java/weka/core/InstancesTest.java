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
 * Copyright (C) 2010 University of Waikato, Hamilton, NZ
 */

package weka.core;

import weka.core.converters.ConverterUtils.DataSource;

import java.util.ArrayList;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import junit.textui.TestRunner;

/**
 * Tests Instances. Run from the command line with:<p/>
 * java weka.core.InstancesTest
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 6318 $
 */
public class InstancesTest
  extends TestCase {

  /** the test instances to work with. */
  protected Instances m_Instances;

  /**
   * Constructs the <code>InstancesTest</code>.
   *
   * @param name 	the name of the test
   */
  public InstancesTest(String name) {
    super(name);
  }

  /**
   * Called by JUnit before each test method.
   *
   * @throws Exception 	if an error occurs
   */
  protected void setUp() throws Exception {
    super.setUp();

    m_Instances = DataSource.read(ClassLoader.getSystemResourceAsStream("weka/core/data/InstancesTest.arff"));
  }

  /**
   * Called by JUnit after each test method.
   *
   * @throws Exception 	if an error occurs
   */
  protected void tearDown() throws Exception {
    m_Instances = null;

    super.tearDown();
  }

  /**
   * Returns the test suite.
   *
   * @return		the test suite
   */
  public static Test suite() {
    return new TestSuite(InstancesTest.class);
  }

  /**
   * Tests the creation of a dataset (unique attribute names).
   *
   * @see Instances#Instances(String, ArrayList, int)
   */
  public void testCreationUnique() {
    Instances	data;
    FastVector	atts;
    FastVector	labels;
    String	relName;

    relName = "testCreationUnique";
    atts    = new FastVector();
    atts.addElement(new Attribute("att-numeric_1"));
    atts.addElement(new Attribute("att-numeric_2"));
    atts.addElement(new Attribute("att-data_1", "yyyy-MM-dd HH:mm"));
    labels = new FastVector();
    labels.addElement("1");
    labels.addElement("2");
    labels.addElement("3");
    atts.addElement(new Attribute("att-nominal_1", labels));
    labels = new FastVector();
    labels.addElement("yes");
    labels.addElement("no");
    atts.addElement(new Attribute("att-nominal_2", labels));
    atts.addElement(new Attribute("att-string_1", (FastVector) null));
    data = new Instances(relName, atts, 0);

    assertEquals("relation name differs", relName, data.relationName());
    assertEquals("# of attributes differ", atts.size(), data.numAttributes());
  }

  /**
   * Tests the creation of a dataset (ambiguous attribute names).
   *
   * @see Instances#Instances(String, ArrayList, int)
   */
  public void testCreationAmbiguous() {
    Instances	data;
    FastVector	atts;
    FastVector	labels;
    String	relName;

    relName = "testCreationAmbiguous";
    atts    = new FastVector();
    atts.addElement(new Attribute("att-numeric_1"));
    atts.addElement(new Attribute("att-numeric_1"));
    atts.addElement(new Attribute("att-data_1", "yyyy-MM-dd HH:mm"));
    labels = new FastVector();
    labels.addElement("1");
    labels.addElement("2");
    labels.addElement("3");
    atts.addElement(new Attribute("att-nominal_1", labels));
    labels = new FastVector();
    labels.addElement("yes");
    labels.addElement("no");
    atts.addElement(new Attribute("att-nominal_1", labels));
    atts.addElement(new Attribute("att-string_1", (FastVector) null));

    try {
      data = new Instances(relName, atts, 0);
    }
    catch (IllegalArgumentException e) {
      data = null;
    }
    assertNull("dataset created with ambiguous attribute names", data);
  }

  /**
   * Tests the copying of the header of a dataset.
   *
   * @see Instances#Instances(Instances, int)
   */
  public void testHeaderCopy() {
    Instances 	data;

    data = new Instances(m_Instances, 0);
    assertEquals("# of attributes differ", m_Instances.numAttributes(), data.numAttributes());
    assertEquals("class index differs", m_Instances.classIndex(), data.classIndex());
    assertEquals("Unexpected instances", 0, data.numInstances());

    m_Instances.setClassIndex(m_Instances.numAttributes() - 1);
    data = new Instances(m_Instances, 0);
    assertEquals("class index differs", m_Instances.classIndex(), data.classIndex());
  }

  /**
   * Tests the full copy of a dataset.
   *
   * @see Instances#Instances(Instances)
   */
  public void testFullCopy() {
    Instances data;

    data = new Instances(m_Instances);
    assertEquals("# of attributes differ", m_Instances.numAttributes(), data.numAttributes());
    assertEquals("class index differs", m_Instances.classIndex(), data.classIndex());
    assertEquals("# of instances differ", m_Instances.numInstances(), data.numInstances());

    m_Instances.setClassIndex(m_Instances.numAttributes() - 1);
    data = new Instances(m_Instances);
    assertEquals("class index differs", m_Instances.classIndex(), data.classIndex());
  }

  /**
   * Tests the partial copy of a dataset.
   *
   * @see Instances#Instances(Instances, int, int)
   */
  public void testPartialCopy() {
    Instances data;

    data = new Instances(m_Instances, 0, m_Instances.numInstances());
    assertEquals("# of instances differ", m_Instances.numInstances(), data.numInstances());

    data = new Instances(m_Instances, 5, 10);
    assertEquals("# of instances differ", 10, data.numInstances());
  }

  /**
   * Executes the test from command-line.
   *
   * @param args	ignored
   */
  public static void main(String[] args){
    TestRunner.run(suite());
  }
}
