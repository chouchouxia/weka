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
 * Copyright (C) 2009 University of Waikato, Hamilton, New Zealand
 */

package weka.clusterers;

import weka.clusterers.AbstractClustererTest;
import weka.clusterers.Clusterer;

import junit.framework.Test;
import junit.framework.TestSuite;

/**
 * Tests HierarchicalClusterer. Run from the command line with:<p/>
 * java weka.clusterers.HierarchicalClustererTest
 *
 * @author Mark Hall
 * @version $Revision: 6588 $
 */
public class HierarchicalClustererTest
  extends AbstractClustererTest {

  public HierarchicalClustererTest(String name) { 
    super(name);  
  }

  /** Creates a default HierarchicalClusterer */
  public Clusterer getClusterer() {
    return new HierarchicalClusterer();
  }

  public static Test suite() {
    return new TestSuite(HierarchicalClustererTest.class);
  }

  public static void main(String[] args){
    junit.textui.TestRunner.run(suite());
  }
}
