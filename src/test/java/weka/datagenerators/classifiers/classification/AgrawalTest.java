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
 * Copyright (C) 2005 University of Waikato, Hamilton, New Zealand
 */

package weka.datagenerators.classifiers.classification;

import weka.datagenerators.AbstractDataGeneratorTest;
import weka.datagenerators.DataGenerator;

import junit.framework.Test;
import junit.framework.TestSuite;

/**
 * Tests Agrawal. Run from the command line with:<p/>
 * java weka.datagenerators.classifiers.classification.AgrawalTest
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 1.2 $
 */
public class AgrawalTest 
  extends AbstractDataGeneratorTest {

  public AgrawalTest(String name) { 
    super(name);  
  }

  /** Creates a default Agrawal */
  public DataGenerator getGenerator() {
    return new Agrawal();
  }

  public static Test suite() {
    return new TestSuite(AgrawalTest.class);
  }

  public static void main(String[] args){
    junit.textui.TestRunner.run(suite());
  }
}
