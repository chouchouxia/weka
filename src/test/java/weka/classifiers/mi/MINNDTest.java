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
 * Copyright (C) 2006 University of Waikato, Hamilton, New Zealand
 */

package weka.classifiers.mi;

import weka.classifiers.AbstractClassifierTest;
import weka.classifiers.Classifier;

import junit.framework.Test;
import junit.framework.TestSuite;

/**
 * Tests MINND. Run from the command line with:<p/>
 * java weka.classifiers.mi.MINNDTest
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 1.2 $
 */
public class MINNDTest 
  extends AbstractClassifierTest {

  public MINNDTest(String name) { 
    super(name);  
  }

  /** Creates a default MINND */
  public Classifier getClassifier() {
    return new MINND();
  }

  public static Test suite() {
    return new TestSuite(MINNDTest.class);
  }

  public static void main(String[] args){
    junit.textui.TestRunner.run(suite());
  }
}
