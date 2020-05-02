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
 *    GUIChooser.java
 *    Copyright (C) 1999 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.gui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.Image;
import java.awt.LayoutManager;
import java.awt.Point;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.Reader;
import java.lang.reflect.Method;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.Vector;

import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSeparator;
import javax.swing.JTable;
import javax.swing.KeyStroke;

import weka.classifiers.bayes.net.GUI;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Copyright;
import weka.core.Instances;
import weka.core.Memory;
import weka.core.SystemInfo;
import weka.core.Utils;
import weka.core.Version;
import weka.gui.arffviewer.ArffViewer;
import weka.gui.beans.KnowledgeFlow;
import weka.gui.beans.KnowledgeFlowApp;
import weka.gui.boundaryvisualizer.BoundaryVisualizer;
import weka.gui.experiment.Experimenter;
import weka.gui.explorer.Explorer;
import weka.gui.graphvisualizer.GraphVisualizer;
import weka.gui.sql.SqlViewer;
import weka.gui.treevisualizer.Node;
import weka.gui.treevisualizer.NodePlace;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeBuild;
import weka.gui.treevisualizer.TreeVisualizer;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;
import weka.gui.visualize.VisualizePanel;

/**
 * The main class for the Weka GUIChooser. Lets the user choose which GUI they
 * want to run.
 * 
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @author Mark Hall (mhall@cs.waikato.ac.nz)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 10793 $
 */
public class GUIChooser extends JFrame {

  static {
    try {
      Object MacApp = Class.forName("com.apple.eawt.Application").newInstance();

      Object macArffHandler =
        Class.forName("weka.gui.MacArffOpenFilesHandler").newInstance();

      Class fileHandlerClass = Class.forName("com.apple.eawt.OpenFilesHandler");
      Class[] paramClass = new Class[1];
      paramClass[0] = fileHandlerClass;
      Object[] args = new Object[1];
      args[0] = macArffHandler;

      Method m = MacApp.getClass().getMethod("setOpenFileHandler", paramClass);
      System.out.println("Trying to install a file handler for Mac...");
      m.invoke(MacApp, macArffHandler);

    } catch (Exception ex) {
      // quietly ignore...
    }
  }

  /** for serialization */
  private static final long serialVersionUID = 9001529425230247914L;

  /** the GUIChooser itself */
  protected GUIChooser m_Self;

  // Menu stuff
  private JMenuBar m_jMenuBar;
  private JMenu m_jMenuProgram;
  private JMenu m_jMenuVisualization;
  private JMenu m_jMenuTools;
  private JMenu m_jMenuHelp;

  // Applications

  /** the panel for the application buttons */
  protected JPanel m_PanelApplications = new JPanel();

  /** Click to open the Explorer */
  protected JButton m_ExplorerBut = new JButton(Messages.getInstance()
    .getString("GUIChooser_Explorer_Text"));

  /** The frame containing the explorer interface */
  protected JFrame m_ExplorerFrame;

  /** Click to open the Explorer */
  protected JButton m_ExperimenterBut = new JButton(Messages.getInstance()
    .getString("GUIChooser_Experimenter_Text"));

  /** The frame containing the experiment interface */
  protected JFrame m_ExperimenterFrame;

  /** Click to open the KnowledgeFlow */
  protected JButton m_KnowledgeFlowBut = new JButton(Messages.getInstance()
    .getString("GUIChooser_KnowledgeFlow_Text"));

  /** Pending file to load on startup of the KnowledgeFlow */
  protected String m_pendingKnowledgeFlowLoad = null;

  /** The frame containing the knowledge flow interface */
  protected JFrame m_KnowledgeFlowFrame;

  /** Click to open the simplecli */
  protected JButton m_SimpleBut = new JButton(Messages.getInstance().getString(
    "GUIChooser_Simple_CLI_Text"));

  /** The SimpleCLI */
  protected SimpleCLI m_SimpleCLI;

  /** keeps track of the opened ArffViewer instancs */
  protected Vector m_ArffViewers = new Vector();

  /** The frame containing the SqlViewer */
  protected JFrame m_SqlViewerFrame;

  /** The frame containing the Bayes net GUI */
  protected JFrame m_BayesNetGUIFrame;

  /** The frame containing the ensemble library interface */
  protected JFrame m_EnsembleLibraryFrame;

  // Visualization

  /** keeps track of the opened plots */
  protected Vector m_Plots = new Vector();

  /** keeps track of the opened ROCs */
  protected Vector m_ROCs = new Vector();

  /** keeps track of the opened tree visualizer instancs */
  protected Vector m_TreeVisualizers = new Vector();

  /** keeps track of the opened graph visualizer instancs */
  protected Vector m_GraphVisualizers = new Vector();

  /** The frame containing the boundary visualizer */
  protected JFrame m_BoundaryVisualizerFrame;

  // Help

  /** The frame containing the system info */
  protected JFrame m_SystemInfoFrame;

  // Other

  /** The frame containing the memory usage */
  protected JFrame m_MemoryUsageFrame;

  /** The frame of the LogWindow */
  protected static LogWindow m_LogWindow = new LogWindow();

  /** The weka image */
  Image m_weka = Toolkit.getDefaultToolkit().getImage(
    GUIChooser.class.getClassLoader().getResource(
      "weka/gui/images/weka_background.gif"));

  /** filechooser for the TreeVisualizer */
  protected JFileChooser m_FileChooserTreeVisualizer = new JFileChooser(
    new File(System.getProperty("user.dir")));

  /** filechooser for the GraphVisualizer */
  protected JFileChooser m_FileChooserGraphVisualizer = new JFileChooser(
    new File(System.getProperty("user.dir")));

  /** filechooser for Plots */
  protected JFileChooser m_FileChooserPlot = new JFileChooser(new File(
    System.getProperty("user.dir")));

  /** filechooser for ROC curves */
  protected JFileChooser m_FileChooserROC = new JFileChooser(new File(
    System.getProperty("user.dir")));

  /** the icon for the frames */
  protected Image m_Icon;

  /** contains the child frames (title &lt;-&gt; object). */
  protected HashSet<Container> m_ChildFrames = new HashSet<Container>();

  /**
   * Create a singleton instance of the GUIChooser
   */
  public static synchronized void createSingleton() {
    if (m_chooser == null) {
      m_chooser = new GUIChooser();
    }
  }

  /**
   * Get the singleton instance of the GUIChooser
   * 
   * @return the singleton instance of the GUIChooser
   */
  public static GUIChooser getSingleton() {
    return m_chooser;
  }

  /**
   * Creates the experiment environment gui with no initial experiment
   */
  public GUIChooser() {

    super(Messages.getInstance().getString("GUIChooser_Title_Text"));

    m_Self = this;

    // filechoosers
    m_FileChooserGraphVisualizer
      .addChoosableFileFilter(new ExtensionFileFilter(".bif", Messages
        .getInstance().getString("GUIChooser_BIF_Files_Text")));
    m_FileChooserGraphVisualizer
      .addChoosableFileFilter(new ExtensionFileFilter(".xml", Messages
        .getInstance().getString("GUIChooser_XML_Files_Text")));

    m_FileChooserPlot.addChoosableFileFilter(new ExtensionFileFilter(
      Instances.FILE_EXTENSION, Messages.getInstance().getString(
        "GUIChooser_ARFF_Files_Text_Front")
        + Instances.FILE_EXTENSION
        + Messages.getInstance().getString("GUIChooser_ARFF_Files_Text_End")));
    m_FileChooserPlot.setMultiSelectionEnabled(true);

    m_FileChooserROC.addChoosableFileFilter(new ExtensionFileFilter(
      Instances.FILE_EXTENSION, Messages.getInstance().getString(
        "GUIChooser_ARFF_Files_Text_Front")
        + Instances.FILE_EXTENSION
        + Messages.getInstance().getString("GUIChooser_ARFF_Files_Text_End")));

    // general layout
    m_Icon =
      Toolkit.getDefaultToolkit().getImage(
        GUIChooser.class.getClassLoader().getResource(
          "weka/gui/weka_icon_new_48.png"));
    setIconImage(m_Icon);
    this.getContentPane().setLayout(new BorderLayout());

    this.getContentPane().add(m_PanelApplications, BorderLayout.EAST);

    // applications
    m_PanelApplications.setBorder(BorderFactory.createTitledBorder(Messages
      .getInstance().getString("GUIChooser_Applications_Text")));
    m_PanelApplications.setLayout(new GridLayout(4, 1));
    m_PanelApplications.add(m_ExplorerBut);
    m_PanelApplications.add(m_ExperimenterBut);
    m_PanelApplications.add(m_KnowledgeFlowBut);
    m_PanelApplications.add(m_SimpleBut);

    // Weka image plus copyright info
    JPanel wekaPan = new JPanel();
    wekaPan.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
    wekaPan.setLayout(new BorderLayout());
    wekaPan.setToolTipText(Messages.getInstance().getString(
      "GUIChooser_Weka_Native_Bird_Text"));
    ImageIcon wii = new ImageIcon(m_weka);
    JLabel wekaLab = new JLabel(wii);
    wekaPan.add(wekaLab, BorderLayout.CENTER);
    String infoString =
      Messages.getInstance().getString("GUIChooser_Information_Text_Front")
        + " " + Version.VERSION + "<br>" + "(c) " + Copyright.getFromYear()
        + " - " + Copyright.getToYear() + "<br>" + Copyright.getOwner()
        + "<br>" + Copyright.getAddress()
        + Messages.getInstance().getString("GUIChooser_Information_Text_End");

    JLabel infoLab = new JLabel(infoString);
    infoLab.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
    wekaPan.add(infoLab, BorderLayout.SOUTH);

    this.getContentPane().add(wekaPan, BorderLayout.CENTER);

    // Menu bar
    m_jMenuBar = new JMenuBar();

    // Program
    m_jMenuProgram = new JMenu();
    m_jMenuBar.add(m_jMenuProgram);
    m_jMenuProgram.setText(Messages.getInstance().getString(
      "GUIChooser_Program_Text"));
    m_jMenuProgram.setMnemonic('P');

    // Program/LogWindow
    JMenuItem jMenuItemProgramLogWindow = new JMenuItem();
    m_jMenuProgram.add(jMenuItemProgramLogWindow);
    jMenuItemProgramLogWindow.setText(Messages.getInstance().getString(
      "GUIChooser_LogWindow_Text"));
    // jMenuItemProgramLogWindow.setMnemonic('L');
    jMenuItemProgramLogWindow.setAccelerator(KeyStroke.getKeyStroke(
      KeyEvent.VK_L, KeyEvent.CTRL_MASK));
    m_LogWindow.setIconImage(m_Icon);
    jMenuItemProgramLogWindow.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        m_LogWindow.setVisible(true);
      }
    });

    final JMenuItem jMenuItemProgramMemUsage = new JMenuItem();
    m_jMenuProgram.add(jMenuItemProgramMemUsage);
    jMenuItemProgramMemUsage.setText(Messages.getInstance().getString(
      "GUIChooser_Memory_Usage_Text"));
    // jMenuItemProgramMemUsage.setMnemonic('M');
    jMenuItemProgramMemUsage.setAccelerator(KeyStroke.getKeyStroke(
      KeyEvent.VK_M, KeyEvent.CTRL_MASK));

    jMenuItemProgramMemUsage.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        if (m_MemoryUsageFrame == null) {
          final MemoryUsagePanel panel = new MemoryUsagePanel();
          jMenuItemProgramMemUsage.setEnabled(false);
          m_MemoryUsageFrame =
            new JFrame(Messages.getInstance().getString(
              "GUIChooser_Memory_Usage_List_Text"));
          m_MemoryUsageFrame.setIconImage(m_Icon);
          m_MemoryUsageFrame.getContentPane().setLayout(new BorderLayout());
          m_MemoryUsageFrame.getContentPane().add(panel, BorderLayout.CENTER);
          m_MemoryUsageFrame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent w) {
              panel.stopMonitoring();
              m_MemoryUsageFrame.dispose();
              m_MemoryUsageFrame = null;
              jMenuItemProgramMemUsage.setEnabled(true);
              checkExit();
            }
          });
          m_MemoryUsageFrame.pack();
          m_MemoryUsageFrame.setSize(400, 50);
          Point l = panel.getFrameLocation();
          if ((l.x != -1) && (l.y != -1)) {
            m_MemoryUsageFrame.setLocation(l);
          }
          m_MemoryUsageFrame.setVisible(true);
          Dimension size = m_MemoryUsageFrame.getPreferredSize();
          m_MemoryUsageFrame.setSize(new Dimension((int) size.getWidth(),
            (int) size.getHeight()));
        }
      }
    });

    m_jMenuProgram.add(new JSeparator());

    // Program/Exit
    JMenuItem jMenuItemProgramExit = new JMenuItem();
    m_jMenuProgram.add(jMenuItemProgramExit);
    jMenuItemProgramExit.setText(Messages.getInstance().getString(
      "GUIChooser_Exit_Text"));
    // jMenuItemProgramExit.setMnemonic('E');
    jMenuItemProgramExit.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_E,
      KeyEvent.CTRL_MASK));
    jMenuItemProgramExit.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        dispose();
        checkExit();
      }
    });

    // Visualization
    m_jMenuVisualization = new JMenu();
    m_jMenuBar.add(m_jMenuVisualization);
    m_jMenuVisualization.setText(Messages
      .getString("GUIChooser_Visualization_Text")); // "Visualization"
    m_jMenuVisualization.setMnemonic('V');

    // Visualization/Plot
    JMenuItem jMenuItemVisualizationPlot = new JMenuItem();
    m_jMenuVisualization.add(jMenuItemVisualizationPlot);
    jMenuItemVisualizationPlot.setText(Messages.getInstance().getString(
      "GUIChooser_Plot_Text"));
    // jMenuItemVisualizationPlot.setMnemonic('P');
    jMenuItemVisualizationPlot.setAccelerator(KeyStroke.getKeyStroke(
      KeyEvent.VK_P, KeyEvent.CTRL_MASK));

    jMenuItemVisualizationPlot.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        // choose file
        int retVal = m_FileChooserPlot.showOpenDialog(m_Self);
        if (retVal != JFileChooser.APPROVE_OPTION) {
          return;
        }

        // build plot
        VisualizePanel panel = new VisualizePanel();
        String filenames = "";
        File[] files = m_FileChooserPlot.getSelectedFiles();
        for (int j = 0; j < files.length; j++) {
          String filename = files[j].getAbsolutePath();
          if (j > 0) {
            filenames += ", ";
          }
          filenames += filename;
          System.err.println(Messages.getInstance().getString(
            "GUIChooser_LoadingInstancesFrom_Error_Text")
            + filename);

          try {
            Reader r = new java.io.BufferedReader(new FileReader(filename));
            Instances i = new Instances(r);
            i.setClassIndex(i.numAttributes() - 1);
            PlotData2D pd1 = new PlotData2D(i);

            if (j == 0) {
              pd1.setPlotName(Messages.getInstance().getString(
                "GUIChooser_MasterPlot_Text"));
              panel.setMasterPlot(pd1);
            } else {
              pd1.setPlotName(Messages.getInstance().getString(
                "GUIChooser_Plot_Text")
                + (j + 1));
              pd1.m_useCustomColour = true;
              pd1.m_customColour = (j % 2 == 0) ? Color.red : Color.blue;
              panel.addPlot(pd1);
            }
          } catch (Exception ex) {
            ex.printStackTrace();
            JOptionPane.showMessageDialog(
              m_Self,
              Messages.getInstance().getString(
                "GUIChooser_Plot_LoadingFile_Error_Text_Front")
                + files[j]
                + Messages.getInstance().getString(
                  "GUIChooser_Plot_LoadingFile_Error_Text_End")
                + ex.getMessage());
            return;
          }
        }

        // create frame
        final JFrame frame =
          new JFrame(Messages.getInstance().getString(
            "GUIChooser_Plot_PlotName_Text")
            + filenames);
        frame.setIconImage(m_Icon);
        frame.getContentPane().setLayout(new BorderLayout());
        frame.getContentPane().add(panel, BorderLayout.CENTER);
        frame.addWindowListener(new WindowAdapter() {
          @Override
          public void windowClosing(WindowEvent e) {
            m_Plots.remove(frame);
            frame.dispose();
            checkExit();
          }
        });
        frame.pack();
        frame.setSize(800, 600);
        frame.setVisible(true);
        m_Plots.add(frame);
      }
    });

    // Visualization/ROC
    JMenuItem jMenuItemVisualizationROC = new JMenuItem();
    m_jMenuVisualization.add(jMenuItemVisualizationROC);
    jMenuItemVisualizationROC.setText(Messages.getInstance().getString(
      "GUIChooser_ROC_Text"));
    // jMenuItemVisualizationROC.setMnemonic('R');
    jMenuItemVisualizationROC.setAccelerator(KeyStroke.getKeyStroke(
      KeyEvent.VK_R, KeyEvent.CTRL_MASK));

    jMenuItemVisualizationROC.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        // choose file
        int retVal = m_FileChooserROC.showOpenDialog(m_Self);
        if (retVal != JFileChooser.APPROVE_OPTION) {
          return;
        }

        // create plot
        String filename = m_FileChooserROC.getSelectedFile().getAbsolutePath();
        Instances result = null;
        try {
          result = new Instances(new BufferedReader(new FileReader(filename)));
        } catch (Exception ex) {
          ex.printStackTrace();
          JOptionPane.showMessageDialog(
            m_Self,
            Messages.getInstance().getString(
              "GUIChooser_ROC_LoadingFile_Error_Text_Front")
              + filename
              + Messages.getInstance().getString(
                "GUIChooser_ROC_LoadingFile_Error_Text_End") + ex.getMessage());
          return;
        }
        result.setClassIndex(result.numAttributes() - 1);
        ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
        vmc.setROCString(Messages.getInstance().getString(
          "GUIChooser_ROC_AreaUnderROC_Text_Front")
          + Utils.doubleToString(ThresholdCurve.getROCArea(result), 4)
          + Messages.getInstance().getString(
            "GUIChooser_ROC_AreaUnderROC_Text_End"));
        vmc.setName(result.relationName());
        PlotData2D tempd = new PlotData2D(result);
        tempd.setPlotName(result.relationName());
        tempd.addInstanceNumberAttribute();
        try {
          vmc.addPlot(tempd);
        } catch (Exception ex) {
          ex.printStackTrace();
          JOptionPane.showMessageDialog(
            m_Self,
            Messages.getInstance().getString(
              "GUIChooser_ROC_AddingPlot_Error_Text")
              + ex.getMessage());
          return;
        }

        final JFrame frame =
          new JFrame(Messages.getInstance().getString(
            "GUIChooser_ROC_File_Text")
            + filename);
        frame.setIconImage(m_Icon);
        frame.getContentPane().setLayout(new BorderLayout());
        frame.getContentPane().add(vmc, BorderLayout.CENTER);
        frame.addWindowListener(new WindowAdapter() {
          @Override
          public void windowClosing(WindowEvent e) {
            m_ROCs.remove(frame);
            frame.dispose();
            checkExit();
          }
        });
        frame.pack();
        frame.setSize(800, 600);
        frame.setVisible(true);
        m_ROCs.add(frame);
      }
    });

    // Visualization/TreeVisualizer
    JMenuItem jMenuItemVisualizationTree = new JMenuItem();
    m_jMenuVisualization.add(jMenuItemVisualizationTree);
    jMenuItemVisualizationTree.setText(Messages.getInstance().getString(
      "GUIChooser_TreeVisualizer_Text"));
    // jMenuItemVisualizationTree.setMnemonic('T');
    jMenuItemVisualizationTree.setAccelerator(KeyStroke.getKeyStroke(
      KeyEvent.VK_T, KeyEvent.CTRL_MASK));

    jMenuItemVisualizationTree.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        // choose file
        int retVal = m_FileChooserTreeVisualizer.showOpenDialog(m_Self);
        if (retVal != JFileChooser.APPROVE_OPTION) {
          return;
        }

        // build tree
        String filename =
          m_FileChooserTreeVisualizer.getSelectedFile().getAbsolutePath();
        TreeBuild builder = new TreeBuild();
        Node top = null;
        NodePlace arrange = new PlaceNode2();
        try {
          top = builder.create(new FileReader(filename));
        } catch (Exception ex) {
          ex.printStackTrace();
          JOptionPane.showMessageDialog(
            m_Self,
            Messages.getInstance().getString(
              "GUIChooser_TreeVisualizer_LoadingFile_Error_Text_Front")
              + filename
              + Messages.getInstance().getString(
                "GUIChooser_TreeVisualizer_LoadingFile_Error_Text_End")
              + ex.getMessage());
          return;
        }

        // create frame
        final JFrame frame =
          new JFrame(Messages.getInstance().getString(
            "GUIChooser_TreeVisualizer_File_Text")
            + filename);
        frame.setIconImage(m_Icon);
        frame.getContentPane().setLayout(new BorderLayout());
        frame.getContentPane().add(new TreeVisualizer(null, top, arrange),
          BorderLayout.CENTER);
        frame.addWindowListener(new WindowAdapter() {
          @Override
          public void windowClosing(WindowEvent e) {
            m_TreeVisualizers.remove(frame);
            frame.dispose();
            checkExit();
          }
        });
        frame.pack();
        frame.setSize(800, 600);
        frame.setVisible(true);
        m_TreeVisualizers.add(frame);
      }
    });

    // Visualization/GraphVisualizer
    JMenuItem jMenuItemVisualizationGraph = new JMenuItem();
    m_jMenuVisualization.add(jMenuItemVisualizationGraph);
    jMenuItemVisualizationGraph.setText(Messages.getInstance().getString(
      "GUIChooser_GraphVisualizer_Text"));
    // jMenuItemVisualizationGraph.setMnemonic('G');
    jMenuItemVisualizationGraph.setAccelerator(KeyStroke.getKeyStroke(
      KeyEvent.VK_G, KeyEvent.CTRL_MASK));

    jMenuItemVisualizationGraph.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        // choose file
        int retVal = m_FileChooserGraphVisualizer.showOpenDialog(m_Self);
        if (retVal != JFileChooser.APPROVE_OPTION) {
          return;
        }

        // build graph
        String filename =
          m_FileChooserGraphVisualizer.getSelectedFile().getAbsolutePath();
        GraphVisualizer panel = new GraphVisualizer();
        try {
          if (filename.toLowerCase().endsWith(".xml")
            || filename.toLowerCase().endsWith(".bif")) {
            panel.readBIF(new FileInputStream(filename));
          } else {
            panel.readDOT(new FileReader(filename));
          }
        } catch (Exception ex) {
          ex.printStackTrace();
          JOptionPane.showMessageDialog(
            m_Self,
            Messages.getInstance().getString(
              "GUIChooser_GraphVisualizer_LoadingFile_Error_Text_Front")
              + filename
              + Messages.getInstance().getString(
                "GUIChooser_GraphVisualizer_LoadingFile_Error_Text_End")
              + ex.getMessage());
          return;
        }

        // create frame
        final JFrame frame =
          new JFrame(Messages.getInstance().getString(
            "GUIChooser_GraphVisualizer_File_Text")
            + filename);
        frame.setIconImage(m_Icon);
        frame.getContentPane().setLayout(new BorderLayout());
        frame.getContentPane().add(panel, BorderLayout.CENTER);
        frame.addWindowListener(new WindowAdapter() {
          @Override
          public void windowClosing(WindowEvent e) {
            m_GraphVisualizers.remove(frame);
            frame.dispose();
            checkExit();
          }
        });
        frame.pack();
        frame.setSize(800, 600);
        frame.setVisible(true);
        m_GraphVisualizers.add(frame);
      }
    });

    // Visualization/BoundaryVisualizer
    final JMenuItem jMenuItemVisualizationBoundary = new JMenuItem();
    m_jMenuVisualization.add(jMenuItemVisualizationBoundary);
    jMenuItemVisualizationBoundary.setText(Messages.getInstance().getString(
      "GUIChooser_BoundaryVisualizer_Text"));
    // jMenuItemVisualizationBoundary.setMnemonic('B');
    jMenuItemVisualizationBoundary.setAccelerator(KeyStroke.getKeyStroke(
      KeyEvent.VK_B, KeyEvent.CTRL_MASK));

    jMenuItemVisualizationBoundary.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        if (m_BoundaryVisualizerFrame == null) {
          jMenuItemVisualizationBoundary.setEnabled(false);
          m_BoundaryVisualizerFrame =
            new JFrame(Messages.getInstance().getString(
              "GUIChooser_BoundaryVisualizer_JFrame_Text"));
          m_BoundaryVisualizerFrame.setIconImage(m_Icon);
          m_BoundaryVisualizerFrame.getContentPane().setLayout(
            new BorderLayout());

          final BoundaryVisualizer bv = new BoundaryVisualizer();
          m_BoundaryVisualizerFrame.getContentPane().add(bv,
            BorderLayout.CENTER);
          m_BoundaryVisualizerFrame.setSize(bv.getMinimumSize());
          m_BoundaryVisualizerFrame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent w) {
              bv.stopPlotting();
              m_BoundaryVisualizerFrame.dispose();
              m_BoundaryVisualizerFrame = null;
              jMenuItemVisualizationBoundary.setEnabled(true);
              checkExit();
            }
          });
          m_BoundaryVisualizerFrame.pack();
          // m_BoundaryVisualizerFrame.setSize(800, 600);
          m_BoundaryVisualizerFrame.setResizable(false);
          m_BoundaryVisualizerFrame.setVisible(true);
          // dont' do a System.exit after last window got closed!
          BoundaryVisualizer.setExitIfNoWindowsOpen(false);
        }
      }
    });

    // Extensions
    JMenu jMenuExtensions =
      new JMenu(Messages.getInstance().getString("GUIChooser_Extensions_Text"));
    jMenuExtensions.setMnemonic(java.awt.event.KeyEvent.VK_E);
    m_jMenuBar.add(jMenuExtensions);
    jMenuExtensions.setVisible(false);

    String extensions =
      GenericObjectEditor.EDITOR_PROPERTIES.getProperty(
        MainMenuExtension.class.getName(), "");

    if (extensions.length() > 0) {
      jMenuExtensions.setVisible(true);
      String[] classnames =
        GenericObjectEditor.EDITOR_PROPERTIES.getProperty(
          MainMenuExtension.class.getName(), "").split(",");
      Hashtable<String, JMenu> submenus = new Hashtable<String, JMenu>();

      // add all extensions
      for (String classname : classnames) {
        try {
          MainMenuExtension ext =
            (MainMenuExtension) Class.forName(classname).newInstance();

          // menuitem in a submenu?
          JMenu submenu = null;
          if (ext.getSubmenuTitle() != null) {
            submenu = submenus.get(ext.getSubmenuTitle());
            if (submenu == null) {
              submenu = new JMenu(ext.getSubmenuTitle());
              submenus.put(ext.getSubmenuTitle(), submenu);
              insertMenuItem(jMenuExtensions, submenu);
            }
          }

          // create menu item
          JMenuItem menuitem = new JMenuItem();
          menuitem.setText(ext.getMenuTitle());
          // does the extension need a frame or does it have its own
          // ActionListener?
          ActionListener listener = ext.getActionListener(m_Self);
          if (listener != null) {
            menuitem.addActionListener(listener);
          } else {
            final JMenuItem finalMenuitem = menuitem;
            final MainMenuExtension finalExt = ext;
            menuitem.addActionListener(new ActionListener() {
              @Override
              public void actionPerformed(ActionEvent e) {
                Component frame =
                  createFrame(m_Self, finalMenuitem.getText(), null, null,
                    null, -1, -1, null, false, false);
                finalExt.fillFrame(frame);
                frame.setVisible(true);
              }
            });
          }

          // sorted insert of menu item
          if (submenu != null) {
            insertMenuItem(submenu, menuitem);
          } else {
            insertMenuItem(jMenuExtensions, menuitem);
          }
        } catch (Exception e) {
          e.printStackTrace();
        }
      }
    }

    // Tools
    m_jMenuTools = new JMenu();
    m_jMenuBar.add(m_jMenuTools);
    m_jMenuTools.setText(Messages.getInstance().getString(
      "GUIChooser_Tools_Text"));
    m_jMenuTools.setMnemonic('T');

    // Tools/ArffViewer
    JMenuItem jMenuItemToolsArffViewer = new JMenuItem();
    m_jMenuTools.add(jMenuItemToolsArffViewer);
    jMenuItemToolsArffViewer.setText(Messages.getInstance().getString(
      "GUIChooser_ArffViewer_Text"));
    // jMenuItemToolsArffViewer.setMnemonic('A');
    jMenuItemToolsArffViewer.setAccelerator(KeyStroke.getKeyStroke(
      KeyEvent.VK_A, KeyEvent.CTRL_MASK));

    jMenuItemToolsArffViewer.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        final ArffViewer av = new ArffViewer();
        av.addWindowListener(new WindowAdapter() {
          @Override
          public void windowClosing(WindowEvent w) {
            m_ArffViewers.remove(av);
            checkExit();
          }
        });
        av.setVisible(true);
        m_ArffViewers.add(av);
      }
    });

    // Tools/SqlViewer
    final JMenuItem jMenuItemToolsSql = new JMenuItem();
    m_jMenuTools.add(jMenuItemToolsSql);
    jMenuItemToolsSql.setText(Messages.getInstance().getString(
      "GUIChooser_SqlViewer_Text"));
    // jMenuItemToolsSql.setMnemonic('S');
    jMenuItemToolsSql.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_S,
      KeyEvent.CTRL_MASK));

    jMenuItemToolsSql.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        if (m_SqlViewerFrame == null) {
          jMenuItemToolsSql.setEnabled(false);
          final SqlViewer sql = new SqlViewer(null);
          m_SqlViewerFrame =
            new JFrame(Messages.getInstance().getString(
              "GUIChooser_SqlViewer_JFrame_Text"));
          m_SqlViewerFrame.setIconImage(m_Icon);
          m_SqlViewerFrame.getContentPane().setLayout(new BorderLayout());
          m_SqlViewerFrame.getContentPane().add(sql, BorderLayout.CENTER);
          m_SqlViewerFrame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent w) {
              sql.saveSize();
              m_SqlViewerFrame.dispose();
              m_SqlViewerFrame = null;
              jMenuItemToolsSql.setEnabled(true);
              checkExit();
            }
          });
          m_SqlViewerFrame.pack();
          m_SqlViewerFrame.setVisible(true);
        }
      }
    });

    // Tools/Bayes net editor
    final JMenuItem jMenuItemBayesNet = new JMenuItem();
    m_jMenuTools.add(jMenuItemBayesNet);
    jMenuItemBayesNet.setText(Messages.getInstance().getString(
      "GUIChooser_BayesNetEditor_Text"));
    jMenuItemBayesNet.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_N,
      KeyEvent.CTRL_MASK));
    jMenuItemBayesNet.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        if (m_BayesNetGUIFrame == null) {
          jMenuItemBayesNet.setEnabled(false);
          final GUI bayesNetGUI = new GUI();
          JMenuBar bayesBar = bayesNetGUI.getMenuBar();
          m_BayesNetGUIFrame =
            new JFrame(Messages.getInstance().getString(
              "GUIChooser_BayesNetworkEditor_JFrame_Text"));
          m_BayesNetGUIFrame.setIconImage(m_Icon);
          m_BayesNetGUIFrame.setJMenuBar(bayesBar);
          m_BayesNetGUIFrame.getContentPane().add(bayesNetGUI,
            BorderLayout.CENTER);
          m_BayesNetGUIFrame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent w) {
              m_BayesNetGUIFrame.dispose();
              m_BayesNetGUIFrame = null;
              jMenuItemBayesNet.setEnabled(true);
              checkExit();
            }
          });
          m_BayesNetGUIFrame.setSize(800, 600);
          m_BayesNetGUIFrame.setVisible(true);
        }
      }
    });

    // Help
    m_jMenuHelp = new JMenu();
    m_jMenuBar.add(m_jMenuHelp);
    m_jMenuHelp.setText(Messages.getInstance()
      .getString("GUIChooser_Help_Text"));
    m_jMenuHelp.setMnemonic('H');

    // Help/Homepage
    JMenuItem jMenuItemHelpHomepage = new JMenuItem();
    m_jMenuHelp.add(jMenuItemHelpHomepage);
    jMenuItemHelpHomepage.setText(Messages.getInstance().getString(
      "GUIChooser_WekaHomepage_Text"));
    // jMenuItemHelpHomepage.setMnemonic('H');
    jMenuItemHelpHomepage.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_H,
      KeyEvent.CTRL_MASK));
    jMenuItemHelpHomepage.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        BrowserHelper.openURL(Messages.getInstance().getString(
          "GUIChooser_WekaHomepage_URL_Text"));
      }
    });

    m_jMenuHelp.add(new JSeparator());

    // Help/WekaWiki
    JMenuItem jMenuItemHelpWekaWiki = new JMenuItem();
    m_jMenuHelp.add(jMenuItemHelpWekaWiki);
    jMenuItemHelpWekaWiki.setText(Messages.getInstance().getString(
      "GUIChooser_WekaHOWTO_Text"));
    // jMenuItemHelpWekaWiki.setMnemonic('W');
    jMenuItemHelpWekaWiki.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_W,
      KeyEvent.CTRL_MASK));

    jMenuItemHelpWekaWiki.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        BrowserHelper.openURL(Messages.getInstance().getString(
          "GUIChooser_WekaWiki_URL_Text"));
      }
    });

    // Help/Sourceforge
    JMenuItem jMenuItemHelpSourceforge = new JMenuItem();
    m_jMenuHelp.add(jMenuItemHelpSourceforge);
    jMenuItemHelpSourceforge.setText(Messages.getInstance().getString(
      "GUIChooser_WekaOnSourceforge_Text"));
    // jMenuItemHelpSourceforge.setMnemonic('F');
    jMenuItemHelpSourceforge.setAccelerator(KeyStroke.getKeyStroke(
      KeyEvent.VK_F, KeyEvent.CTRL_MASK));

    jMenuItemHelpSourceforge.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        BrowserHelper.openURL(Messages.getInstance().getString(
          "GUIChooser_WekaOnSourceforge_URL_Text"));
      }
    });

    // Help/SystemInfo
    final JMenuItem jMenuItemHelpSysInfo = new JMenuItem();
    m_jMenuHelp.add(jMenuItemHelpSysInfo);
    jMenuItemHelpSysInfo.setText(Messages.getInstance().getString(
      "GUIChooser_SystemInfo_Text"));
    // jMenuItemHelpSysInfo.setMnemonic('S');
    jMenuItemHelpSysInfo.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_I,
      KeyEvent.CTRL_MASK));

    jMenuItemHelpSysInfo.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        if (m_SystemInfoFrame == null) {
          jMenuItemHelpSysInfo.setEnabled(false);
          m_SystemInfoFrame =
            new JFrame(Messages.getInstance().getString(
              "GUIChooser_SystemInfo_JFrame_Text"));
          m_SystemInfoFrame.setIconImage(m_Icon);
          m_SystemInfoFrame.getContentPane().setLayout(new BorderLayout());

          // get info
          Hashtable info = new SystemInfo().getSystemInfo();

          // sort names
          Vector names = new Vector();
          Enumeration enm = info.keys();
          while (enm.hasMoreElements()) {
            names.add(enm.nextElement());
          }
          Collections.sort(names);

          // generate table
          String[][] data = new String[info.size()][2];
          for (int i = 0; i < names.size(); i++) {
            data[i][0] = names.get(i).toString();
            data[i][1] = info.get(data[i][0]).toString();
          }

          String[] titles =
            new String[] {
              Messages.getInstance().getString(
                "GUIChooser_SystemInfo_TitleKey_Text"),
              Messages.getInstance().getString(
                "GUIChooser_SystemInfo_TitleValue_Text") };
          JTable table = new JTable(data, titles);

          m_SystemInfoFrame.getContentPane().add(new JScrollPane(table),
            BorderLayout.CENTER);
          m_SystemInfoFrame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent w) {
              m_SystemInfoFrame.dispose();
              m_SystemInfoFrame = null;
              jMenuItemHelpSysInfo.setEnabled(true);
              checkExit();
            }
          });
          m_SystemInfoFrame.pack();
          m_SystemInfoFrame.setSize(800, 600);
          m_SystemInfoFrame.setVisible(true);
        }
      }
    });

    // applications

    m_ExplorerBut.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        showExplorer(null);
      }
    });

    m_ExperimenterBut.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        if (m_ExperimenterFrame == null) {
          System.out.println("CALLED ExperimentEnvironment");

          m_ExperimenterBut.setEnabled(false);
          m_ExperimenterFrame =
            new JFrame(Messages.getInstance().getString(
              "GUIChooser_WekaExperimentEnvironment_JFrame_Text"));
          m_ExperimenterFrame.setIconImage(m_Icon);
          m_ExperimenterFrame.getContentPane().setLayout(new BorderLayout());
          m_ExperimenterFrame.getContentPane().add(new Experimenter(false),
            BorderLayout.CENTER);
          m_ExperimenterFrame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent w) {
              m_ExperimenterFrame.dispose();
              m_ExperimenterFrame = null;
              m_ExperimenterBut.setEnabled(true);
              checkExit();
            }
          });
          m_ExperimenterFrame.pack();
          m_ExperimenterFrame.setSize(800, 600);
          m_ExperimenterFrame.setVisible(true);
        }
      }
    });

    KnowledgeFlowApp.addStartupListener(new weka.gui.beans.StartUpListener() {
      @Override
      public void startUpComplete() {
        if (m_KnowledgeFlowFrame == null) {
          final KnowledgeFlowApp kna = KnowledgeFlowApp.getSingleton();
          m_KnowledgeFlowBut.setEnabled(false);
          if (m_pendingKnowledgeFlowLoad != null
            && m_pendingKnowledgeFlowLoad.length() > 0) {
            KnowledgeFlowApp.getSingleton().loadInitialLayout(
              m_pendingKnowledgeFlowLoad);
            m_pendingKnowledgeFlowLoad = null;
          }
          m_KnowledgeFlowFrame =
            new JFrame(Messages.getInstance().getString(
              "GUIChooser_WekaKnowledgeFlowEnvironment_JFrame_Text"));
          m_KnowledgeFlowFrame.setIconImage(m_Icon);
          m_KnowledgeFlowFrame.getContentPane().setLayout(new BorderLayout());
          m_KnowledgeFlowFrame.getContentPane().add(kna, BorderLayout.CENTER);
          m_KnowledgeFlowFrame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent w) {
              kna.clearLayout();
              m_KnowledgeFlowFrame.dispose();
              m_KnowledgeFlowFrame = null;
              m_KnowledgeFlowBut.setEnabled(true);
              checkExit();
            }
          });
          m_KnowledgeFlowFrame.pack();
          m_KnowledgeFlowFrame.setSize(1000, 750);
          m_KnowledgeFlowFrame.setVisible(true);
        }
      }
    });

    m_KnowledgeFlowBut.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        showKnowledgeFlow(null);
      }
    });

    m_SimpleBut.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        if (m_SimpleCLI == null) {
          m_SimpleBut.setEnabled(false);
          try {
            m_SimpleCLI = new SimpleCLI();
            m_SimpleCLI.setIconImage(m_Icon);
          } catch (Exception ex) {
            throw new Error(Messages.getInstance().getString(
              "GUIChooser_UnableToStartSimpleCLI_Error_Text"));
          }
          m_SimpleCLI.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent w) {
              m_SimpleCLI.dispose();
              m_SimpleCLI = null;
              m_SimpleBut.setEnabled(true);
              checkExit();
            }
          });
          m_SimpleCLI.setVisible(true);
        }
      }
    });

    /*
     * m_EnsembleLibraryBut.addActionListener(new ActionListener() { public void
     * actionPerformed(ActionEvent e) { if (m_EnsembleLibraryFrame == null) {
     * m_EnsembleLibraryBut.setEnabled(false); m_EnsembleLibraryFrame = new
     * JFrame("EnsembleLibrary"); m_EnsembleLibraryFrame.setIconImage(m_Icon);
     * m_EnsembleLibraryFrame.getContentPane().setLayout(new BorderLayout());
     * EnsembleLibrary value = new EnsembleLibrary(); EnsembleLibraryEditor
     * libraryEditor = new EnsembleLibraryEditor();
     * libraryEditor.setValue(value);
     * m_EnsembleLibraryFrame.getContentPane().add
     * (libraryEditor.getCustomEditor(), BorderLayout.CENTER);
     * m_EnsembleLibraryFrame.addWindowListener(new WindowAdapter() { public
     * void windowClosing(WindowEvent w) { m_EnsembleLibraryFrame.dispose();
     * m_EnsembleLibraryFrame = null; m_EnsembleLibraryBut.setEnabled(true);
     * checkExit(); } }); m_EnsembleLibraryFrame.pack();
     * m_EnsembleLibraryFrame.setSize(800, 600);
     * m_EnsembleLibraryFrame.setVisible(true); } } });
     */

    setJMenuBar(m_jMenuBar);

    addWindowListener(new WindowAdapter() {
      @Override
      public void windowClosing(WindowEvent w) {
        dispose();
        checkExit();
      }
    });
    pack();
  }

  public void showKnowledgeFlow(String fileToLoad) {
    if (m_KnowledgeFlowFrame == null) {
      KnowledgeFlow.startApp();
      m_pendingKnowledgeFlowLoad = fileToLoad;
    } else {
      if (fileToLoad != null) {
        KnowledgeFlowApp.getSingleton().loadInitialLayout(fileToLoad);
      }
    }
  }

  public void showExplorer(String fileToLoad) {
    Explorer expl = null;
    if (m_ExplorerFrame == null) {
      m_ExplorerBut.setEnabled(false);
      m_ExplorerFrame = new JFrame("Weka Explorer");
      m_ExplorerFrame.setIconImage(m_Icon);
      m_ExplorerFrame.getContentPane().setLayout(new BorderLayout());
      expl = new Explorer();

      m_ExplorerFrame.getContentPane().add(expl, BorderLayout.CENTER);
      m_ExplorerFrame.addWindowListener(new WindowAdapter() {
        @Override
        public void windowClosing(WindowEvent w) {
          m_ExplorerFrame.dispose();
          m_ExplorerFrame = null;
          m_ExplorerBut.setEnabled(true);
          checkExit();
        }
      });
      m_ExplorerFrame.pack();
      m_ExplorerFrame.setSize(800, 600);
      m_ExplorerFrame.setVisible(true);
    } else {
      Object o = m_ExplorerFrame.getContentPane().getComponent(0);
      if (o instanceof Explorer) {
        expl = (Explorer) o;
      }
    }

    if (fileToLoad != null) {
      try {
        weka.core.converters.AbstractFileLoader loader =
          weka.core.converters.ConverterUtils.getLoaderForFile(fileToLoad);
        loader.setFile(new File(fileToLoad));
        expl.getPreprocessPanel().setInstancesFromFile(loader);
      } catch (Exception ex) {
        ex.printStackTrace();
      }
    }
  }

  /**
   * insert the menu item in a sorted fashion.
   * 
   * @param menu the menu to add the item to
   * @param menuitem the menu item to add
   */
  protected void insertMenuItem(JMenu menu, JMenuItem menuitem) {
    insertMenuItem(menu, menuitem, 0);
  }

  /**
   * insert the menu item in a sorted fashion.
   * 
   * @param menu the menu to add the item to
   * @param menuitem the menu item to add
   * @param startIndex the index in the menu to start with (0-based)
   */
  protected void insertMenuItem(JMenu menu, JMenuItem menuitem, int startIndex) {
    boolean inserted;
    int i;
    JMenuItem current;
    String currentStr;
    String newStr;

    inserted = false;
    newStr = menuitem.getText().toLowerCase();

    // try to find a spot inbetween
    for (i = startIndex; i < menu.getMenuComponentCount(); i++) {
      if (!(menu.getMenuComponent(i) instanceof JMenuItem)) {
        continue;
      }

      current = (JMenuItem) menu.getMenuComponent(i);
      currentStr = current.getText().toLowerCase();
      if (currentStr.compareTo(newStr) > 0) {
        inserted = true;
        menu.insert(menuitem, i);
        break;
      }
    }

    // add it at the end if not yet inserted
    if (!inserted) {
      menu.add(menuitem);
    }
  }

  /**
   * creates a frame and returns it.
   * 
   * @param parent the parent of the generated frame
   * @param title the title of the frame
   * @param c the component to place, can be null
   * @param layout the layout to use, e.g., BorderLayout
   * @param layoutConstraints the layout constraints, e.g., BorderLayout.CENTER
   * @param width the width of the frame, ignored if -1
   * @param height the height of the frame, ignored if -1
   * @param menu an optional menu
   * @param listener if true a default listener is added
   * @param visible if true then the frame is made visible immediately
   * @return the generated frame
   */
  protected Container createFrame(GUIChooser parent, String title, Component c,
    LayoutManager layout, Object layoutConstraints, int width, int height,
    JMenuBar menu, boolean listener, boolean visible) {

    Container result = null;

    final ChildFrameSDI frame = new ChildFrameSDI(parent, title);

    // layout
    frame.setLayout(layout);
    if (c != null) {
      frame.getContentPane().add(c, layoutConstraints);
    }

    // menu
    frame.setJMenuBar(menu);

    // size
    frame.pack();
    if ((width > -1) && (height > -1)) {
      frame.setSize(width, height);
    }
    frame.validate();

    // location
    int screenHeight = getGraphicsConfiguration().getBounds().height;
    int screenWidth = getGraphicsConfiguration().getBounds().width;
    frame.setLocation((screenWidth - frame.getBounds().width) / 2,
      (screenHeight - frame.getBounds().height) / 2);

    // listener?
    if (listener) {
      frame.addWindowListener(new WindowAdapter() {
        @Override
        public void windowClosing(WindowEvent e) {
          frame.dispose();
        }
      });
    }

    // display frame
    if (visible) {
      frame.setVisible(true);
    }

    result = frame;

    return result;
  }

  /**
   * Specialized JFrame class.
   * 
   * @author fracpete (fracpete at waikato dot ac dot nz)
   * @version $Revision: 10793 $
   */
  public static class ChildFrameSDI extends JFrame {

    /** for serialization. */
    private static final long serialVersionUID = 8588293938686425618L;

    /** the parent frame. */
    protected GUIChooser m_Parent;

    /**
     * constructs a new internal frame that knows about its parent.
     * 
     * @param parent the parent frame
     * @param title the title of the frame
     */
    public ChildFrameSDI(GUIChooser parent, String title) {
      super(title);

      m_Parent = parent;

      addWindowListener(new WindowAdapter() {
        @Override
        public void windowActivated(WindowEvent e) {
          // update title of parent
          if (getParentFrame() != null) {
            getParentFrame().createTitle(getTitle());
          }
        }
      });

      // add to parent
      if (getParentFrame() != null) {
        getParentFrame().addChildFrame(this);
        setIconImage(getParentFrame().getIconImage());
      }
    }

    /**
     * returns the parent frame, can be null.
     * 
     * @return the parent frame
     */
    public GUIChooser getParentFrame() {
      return m_Parent;
    }

    /**
     * de-registers the child frame with the parent first.
     */
    @Override
    public void dispose() {
      if (getParentFrame() != null) {
        getParentFrame().removeChildFrame(this);
        getParentFrame().createTitle("");
      }

      super.dispose();
    }
  }

  /**
   * creates and displays the title.
   * 
   * @param title the additional part of the title
   */
  protected void createTitle(String title) {
    String newTitle;

    newTitle =
      Messages.getInstance().getString("GUIChooser_CreateTitle_Text_Front")
        + new Version();
    if (title.length() != 0) {
      newTitle +=
        Messages.getInstance().getString("GUIChooser_CreateTitle_Text_End")
          + title;
    }

    setTitle(newTitle);
  }

  /**
   * adds the given child frame to the list of frames.
   * 
   * @param c the child frame to add
   */
  public void addChildFrame(Container c) {
    m_ChildFrames.add(c);
  }

  /**
   * tries to remove the child frame, it returns true if it could do such.
   * 
   * @param c the child frame to remove
   * @return true if the child frame could be removed
   */
  public boolean removeChildFrame(Container c) {
    boolean result = m_ChildFrames.remove(c);
    return result;
  }

  /**
   * Kills the JVM if all windows have been closed.
   */
  private void checkExit() {

    if (!isVisible()
      // applications
      && (m_ExplorerFrame == null)
      && (m_ExperimenterFrame == null)
      && (m_KnowledgeFlowFrame == null)
      && (m_SimpleCLI == null)
      // tools
      && (m_ArffViewers.size() == 0)
      && (m_SqlViewerFrame == null)
      && (m_EnsembleLibraryFrame == null)
      // visualization
      && (m_Plots.size() == 0) && (m_ROCs.size() == 0)
      && (m_TreeVisualizers.size() == 0) && (m_GraphVisualizers.size() == 0)
      && (m_BoundaryVisualizerFrame == null)
      // help
      && (m_SystemInfoFrame == null)) {
      System.exit(0);
    }
  }

  /**
   * variable for the GUIChooser class which would be set to null by the memory
   * monitoring thread to free up some memory if we running out of memory
   */
  private static GUIChooser m_chooser;

  /** for monitoring the Memory consumption */
  private static Memory m_Memory = new Memory(true);

  /**
   * Tests out the GUIChooser environment.
   * 
   * @param args ignored.
   */
  public static void main(String[] args) {

    weka.core.logging.Logger.log(weka.core.logging.Logger.Level.INFO, Messages
      .getInstance().getString("GUIChooser_Main_LoggingStarted_Text"));
    LookAndFeel.setLookAndFeel();

    try {

      // uncomment to disable the memory management:
      // m_Memory.setEnabled(false);

      // m_chooser = new GUIChooser();
      GUIChooser.createSingleton();
      m_chooser.setVisible(true);

      if (args != null && args.length > 0) {
        m_chooser.showExplorer(args[0]);
      }

      Thread memMonitor = new Thread() {
        @Override
        public void run() {
          while (true) {
            // try {
            // //System.out.println("before sleeping");
            // this.sleep(4000);

            // System.gc();

            if (m_Memory.isOutOfMemory()) {
              // clean up
              m_chooser.dispose();
              if (m_chooser.m_ExperimenterFrame != null) {
                m_chooser.m_ExperimenterFrame.dispose();
                m_chooser.m_ExperimenterFrame = null;
              }
              if (m_chooser.m_ExplorerFrame != null) {
                m_chooser.m_ExplorerFrame.dispose();
                m_chooser.m_ExplorerFrame = null;
              }
              if (m_chooser.m_KnowledgeFlowFrame != null) {
                m_chooser.m_KnowledgeFlowFrame.dispose();
                m_chooser.m_KnowledgeFlowFrame = null;
              }
              if (m_chooser.m_SimpleCLI != null) {
                m_chooser.m_SimpleCLI.dispose();
                m_chooser.m_SimpleCLI = null;
              }
              if (m_chooser.m_ArffViewers.size() > 0) {
                for (int i = 0; i < m_chooser.m_ArffViewers.size(); i++) {
                  ArffViewer av = (ArffViewer) m_chooser.m_ArffViewers.get(i);
                  av.dispose();
                }
                m_chooser.m_ArffViewers.clear();
              }
              m_chooser = null;
              System.gc();

              // display error
              m_chooser.m_LogWindow.setVisible(true);
              m_chooser.m_LogWindow.toFront();
              System.err.println(Messages.getInstance().getString(
                "GUIChooser_Main_Error_Text_Front"));
              m_Memory.showOutOfMemory();
              System.err.println(Messages.getInstance().getString(
                "GUIChooser_Main_Error_Text_End"));
              System.exit(-1);
            }
            // }
            // catch(InterruptedException ex) {
            // ex.printStackTrace();
            // }
          }
        }
      };

      memMonitor.setPriority(Thread.NORM_PRIORITY);
      memMonitor.start();
    } catch (Exception ex) {
      ex.printStackTrace();
      System.err.println(ex.getMessage());
    }
  }
}
