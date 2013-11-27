
public class Main {

	public static void main(String[] args) {
		System.out.println("Started");
		System.out.println(java.lang.Runtime.getRuntime().totalMemory());
		System.out.println(java.lang.Runtime.getRuntime().maxMemory());
		Main m = new Main();
		m.doWekaOperations();
	}
	
	public void doWekaOperations() {
		WekaOperations w = new WekaOperations();
//		w.classifyJ48Tree();
		w.classifyGeneticProgrammingFunction();
//		w.classifySVM();
//		w.classifyNaiveBayes();
//		w.classifyAll();
//		w.classifyAllV2();
//		w.classifyAllV3WithMultipleRuns();
//		w.classifyIBkV2();
	}
	
}
