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
 * Copyright (C) 2008 University of Waikato, Hamilton, New Zealand
 */

package weka.attributeSelection;

import junit.framework.Test;
import junit.framework.TestSuite;

/**
 * Tests ScatterSearchV1. Run from the command line with:<p/>
 * java weka.attributeSelection.ScatterSearchV1Test
 *
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 * @version $Revision: 5562 $
 */
public class ScatterSearchV1Test 
  extends AbstractSearchTest {

  public ScatterSearchV1Test(String name) { 
    super(name);  
  }

  /** Creates a default GeneticSearch */
  public ASSearch getSearch() {
    return new ScatterSearchV1();
  }

  /** Creates a default CfsSubsetEval */
  public ASEvaluation getEvaluator() {
    return new CfsSubsetEval();
  }

  public static Test suite() {
    return new TestSuite(ScatterSearchV1Test.class);
  }

  public static void main(String[] args){
    junit.textui.TestRunner.run(suite());
  }
}
