# Dataset audit

```json
{
  "rows": 256,
  "columns": [
    {
      "name": "WaterSpeed (m/s)",
      "dtype": "float64",
      "unit_in_header": "m/s"
    },
    {
      "name": "PulseEnergy (microJ)",
      "dtype": "float64",
      "unit_in_header": "microJ"
    },
    {
      "name": "Frequency (kHz)",
      "dtype": "float64",
      "unit_in_header": "kHz"
    },
    {
      "name": "ScanningSpeed (mm/s)",
      "dtype": "float64",
      "unit_in_header": "mm/s"
    },
    {
      "name": "Width (micrometer)",
      "dtype": "float64",
      "unit_in_header": "micrometer"
    },
    {
      "name": "Depth (micrometer)",
      "dtype": "float64",
      "unit_in_header": "micrometer"
    },
    {
      "name": "MRR (cubic micrometer per sec)",
      "dtype": "float64",
      "unit_in_header": "cubic micrometer per sec"
    },
    {
      "name": "HAZ (micrometer)",
      "dtype": "float64",
      "unit_in_header": "micrometer"
    }
  ],
  "missing_values": {
    "WaterSpeed (m/s)": 0,
    "PulseEnergy (microJ)": 0,
    "Frequency (kHz)": 0,
    "ScanningSpeed (mm/s)": 0,
    "Width (micrometer)": 0,
    "Depth (micrometer)": 0,
    "MRR (cubic micrometer per sec)": 0,
    "HAZ (micrometer)": 0
  },
  "duplicate_rows": 0,
  "unique_input_combinations": 256,
  "input_levels": {
    "WaterSpeed (m/s)": 4,
    "PulseEnergy (microJ)": 4,
    "Frequency (kHz)": 4,
    "ScanningSpeed (mm/s)": 4
  },
  "replicates_per_combination": {
    "1": 256
  },
  "design_assessment": "complete factorial-looking grid",
  "negative_values": {
    "Width (micrometer)": 0,
    "Depth (micrometer)": 18,
    "MRR (cubic micrometer per sec)": 22,
    "HAZ (micrometer)": 0
  },
  "ranges": {
    "WaterSpeed (m/s)": {
      "min": 6.0,
      "max": 8.0
    },
    "PulseEnergy (microJ)": {
      "min": 10.0,
      "max": 30.0
    },
    "Frequency (kHz)": {
      "min": 2.0,
      "max": 18.0
    },
    "ScanningSpeed (mm/s)": {
      "min": 1.0,
      "max": 9.0
    },
    "Width (micrometer)": {
      "min": 14.850618,
      "max": 32.550938
    },
    "Depth (micrometer)": {
      "min": -6.633542777777752,
      "max": 39.98669055555559
    },
    "MRR (cubic micrometer per sec)": {
      "min": -21.35033799999999,
      "max": 63.21726955555562
    },
    "HAZ (micrometer)": {
      "min": 25.35317355555556,
      "max": 69.349418
    }
  },
  "correlations": {
    "WaterSpeed (m/s)": {
      "WaterSpeed (m/s)": 1.0,
      "PulseEnergy (microJ)": -3.2474023470285856e-16,
      "Frequency (kHz)": -7.650130529057728e-16,
      "ScanningSpeed (mm/s)": 4.3715031594615573e-16,
      "Width (micrometer)": -0.27286592029638396,
      "Depth (micrometer)": 0.09659915546624222,
      "MRR (cubic micrometer per sec)": 0.14162632366116312,
      "HAZ (micrometer)": -0.27061634473500634
    },
    "PulseEnergy (microJ)": {
      "WaterSpeed (m/s)": -3.2474023470285856e-16,
      "PulseEnergy (microJ)": 1.0,
      "Frequency (kHz)": -7.494005416219807e-17,
      "ScanningSpeed (mm/s)": 1.4613310561628617e-16,
      "Width (micrometer)": 0.5457296863938,
      "Depth (micrometer)": 0.13962685068259187,
      "MRR (cubic micrometer per sec)": 0.4847055912294247,
      "HAZ (micrometer)": 0.5072173958948536
    },
    "Frequency (kHz)": {
      "WaterSpeed (m/s)": -7.650130529057728e-16,
      "PulseEnergy (microJ)": -7.494005416219807e-17,
      "Frequency (kHz)": 1.0,
      "ScanningSpeed (mm/s)": 2.8102520310824274e-17,
      "Width (micrometer)": 0.5170939195523995,
      "Depth (micrometer)": 0.42706333492498827,
      "MRR (cubic micrometer per sec)": 0.7815316020348297,
      "HAZ (micrometer)": 0.6445006295827982
    },
    "ScanningSpeed (mm/s)": {
      "WaterSpeed (m/s)": 4.3715031594615573e-16,
      "PulseEnergy (microJ)": 1.4613310561628617e-16,
      "Frequency (kHz)": 2.8102520310824274e-17,
      "ScanningSpeed (mm/s)": 1.0,
      "Width (micrometer)": -0.452399016236295,
      "Depth (micrometer)": -0.7344030174577338,
      "MRR (cubic micrometer per sec)": 0.1113860739065767,
      "HAZ (micrometer)": -0.2170120196202537
    },
    "Width (micrometer)": {
      "WaterSpeed (m/s)": -0.27286592029638396,
      "PulseEnergy (microJ)": 0.5457296863938,
      "Frequency (kHz)": 0.5170939195523995,
      "ScanningSpeed (mm/s)": -0.452399016236295,
      "Width (micrometer)": 1.0,
      "Depth (micrometer)": 0.6513822884246591,
      "MRR (cubic micrometer per sec)": 0.6003765636262441,
      "HAZ (micrometer)": 0.9154842342498851
    },
    "Depth (micrometer)": {
      "WaterSpeed (m/s)": 0.09659915546624222,
      "PulseEnergy (microJ)": 0.13962685068259187,
      "Frequency (kHz)": 0.42706333492498827,
      "ScanningSpeed (mm/s)": -0.7344030174577338,
      "Width (micrometer)": 0.6513822884246591,
      "Depth (micrometer)": 1.0,
      "MRR (cubic micrometer per sec)": 0.3320237245127808,
      "HAZ (micrometer)": 0.44902097708923006
    },
    "MRR (cubic micrometer per sec)": {
      "WaterSpeed (m/s)": 0.14162632366116312,
      "PulseEnergy (microJ)": 0.4847055912294247,
      "Frequency (kHz)": 0.7815316020348297,
      "ScanningSpeed (mm/s)": 0.1113860739065767,
      "Width (micrometer)": 0.6003765636262441,
      "Depth (micrometer)": 0.3320237245127808,
      "MRR (cubic micrometer per sec)": 1.0,
      "HAZ (micrometer)": 0.6549966591005377
    },
    "HAZ (micrometer)": {
      "WaterSpeed (m/s)": -0.27061634473500634,
      "PulseEnergy (microJ)": 0.5072173958948536,
      "Frequency (kHz)": 0.6445006295827982,
      "ScanningSpeed (mm/s)": -0.2170120196202537,
      "Width (micrometer)": 0.9154842342498851,
      "Depth (micrometer)": 0.44902097708923006,
      "MRR (cubic micrometer per sec)": 0.6549966591005377,
      "HAZ (micrometer)": 1.0
    }
  },
  "open_questions": [
    "Are rows measured experiments, averaged replicates, response-surface values, or synthetic values?",
    "What are the material, pulse duration, wavelength, spot size, focusing, water, instrument, and measurement-procedure metadata?",
    "Are negative depth and MRR values physically valid signed quantities or data-generation artifacts?",
    "Is MRR independently measured or computed from geometry and scanning speed?",
    "Are there hidden run, specimen, batch, or session identifiers?"
  ],
  "mrr_geometry_check": {
    "candidate": "width * depth * scanning speed",
    "r2_with_intercept": 0.7345308651367912,
    "ratio_std": 0.1382491791688015,
    "interpretation": "not verified as an exact derivation; advisor confirmation required"
  }
}
```
