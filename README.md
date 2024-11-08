# Speech & Audio Processing & Recognition
Hello! This is an assignment repository for the University of Surrey's SAPR class regarding speech synthesis and recognition.
The assignment 1 write-up can be found [here](https://www.notion.so/frankcholula/Speech-Synthesis-1233b40fbcd58097888fec180e23754f?pvs=4).

If you are a **University of Surrey student**, you are welcome to use this project as a learning resource and reference for your coursework. A simple credit to the OC (wee! that's me, [Frank](https://frankcholula.notion.site/)) would be greatly appreciated. However, please note that submitting this work as your own academic assignment is not permitted and may lead to [academic misconduct penalties](https://www.surrey.ac.uk/office-student-complaints-appeals-and-regulation/academic-misconduct-and-appeals). Just make sure you're submitting your orignal work.

## Directory Layout
The helper functions can be found in the `assignment_1` directory. All the files ending in `.m` are MATLAB functions and the actual speech synthesis and analysis is done in the `speech_synthesis.mlx` file.

The `speech` directory contains the audio files used in the assignment. The `synthesized_impulse` and `synthesized_sawtooth` directories contain the synthesized audio files.

```bash
assignment_1
├── LICENSE
├── README.md
└── assignment_1
    ├── assignment1_submission.pdf
    ├── assignment1_spec.pdf
    ├── estimateF0ByAutoCorrelation.m
    ├── estimateF0ByPowerSpectrum.m
    ├── estimateFormants.m
    ├── extractCenterSegment.m
    ├── orderAnalysis.m
    ├── orderSegmentAnalysis.m
    ├── playAudio.m
    ├── plotLPCPoleZero.m
    ├── plotLPCResponse.m
    ├── plotSpectrogram.m
    ├── plotSynthesis.m
    ├── segmentAnalysis.m
    ├── speech
    ├── speech_synthesis.mlx
    ├── synthesizeLPC.m
    ├── synthesized_impulse
    └── synthesized_sawtooth
```

## License
This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html) (GPL-3.0). This means:
1.	**Attribution**: You must give appropriate credit to the original author (me, [Frank Lu](https://frankcholula.notion.site/)) if you use or modify this project.
2.	**Non-Proprietary**: Any derivative works or modifications must also be shared under the same license. This project cannot be used in proprietary or closed-source software.
3.	**Open Source Forever**: Any modifications you make must remain open-source under the GPL-3.0 license. This helps ensure the code remains accessible and beneficial to everyone.

You can read the full license in the [LICENSE](LICENSE) file.