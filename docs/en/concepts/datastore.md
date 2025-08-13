# Mental Model of DataStore (`minto.v1.datastore.DataStore`)

## Introduction

As introduced in the [Overview](./overview.md), the `DataStore` is the fundamental building block in Minto for organizing experimental data. It acts as a centralized container holding all relevant information associated with a specific experimental context—either data common to an entire experiment or data specific to a single run within that experiment.

This page delves deeper into the structure and concept of the `DataStore` to help you build a clear mental model of how to use it effectively.

## The Core Idea: A Structured Digital Notebook

Think of a `DataStore` as a **highly structured digital laboratory notebook** specifically designed for computational experiments, particularly in optimization. Instead of unstructured notes or scattered files, a `DataStore` provides predefined categories to systematically store different kinds of information related to your experiment.

Its main goals are:
* **Centralization:** Bring together all data related to one experimental context (instance, parameters, results, metadata) into a single object.
* **Structure:** Organize this data into logical categories, making it easier to find, understand, and process later.
* **Consistency:** Provide a standard format for experimental data, facilitating comparison and sharing.

## Key Data Categories within `DataStore`

A `DataStore` organizes data into several primary attributes, each holding a specific type of information. These attributes are typically populated using the corresponding `log_xxx` functions (covered in the [Logging Guide](../logging_guide.md)).

Here are the main categories:

### 1. Instance Data (`.instance`)

* **Purpose:** Stores the definition of the problem being solved.
* **Content:** This typically includes details about the optimization problem itself, such as decision variables (their types, bounds), constraints, and the objective function structure.
* **Format:** Often represented using a structured format, potentially compatible with libraries like OMMX (`ommx_v1.Instance`). It might contain summary information like the number of variables (binary, integer, continuous), number of constraints, and a title.
* **Logging:** Populated using `log_instance()`.

### 2. Parameters (`.parameters`)

* **Purpose:** Stores the specific settings or parameters used for this particular experiment or run.
* **Content:** This includes algorithm hyperparameters, solver options, time limits, random seeds, or any other configuration details that define *how* the experiment was executed.
* **Format:** Typically a simple dictionary-like structure (key-value pairs).
* **Logging:** Populated using `log_parameter()`.

### 3. Results (`.solution` / `.sampleset`)

* **Purpose:** Stores the outcome(s) of the experimental run. Minto can handle different types of results, commonly including:
    * **`.solution`:** Often used for results from exact solvers or methods that return a single, definitive solution. This might contain the final objective function value, whether the solution is feasible or optimal, variable values, and potentially dual information. (e.g., `ommx_v1.Solution`).
    * **`.sampleset`:** Typically used for results from heuristic methods, sampling algorithms (like simulated annealing or quantum annealing), or any method that produces multiple potential solutions or samples. This usually contains multiple samples, their corresponding objective values (or energies), number of occurrences, and potentially timing information. (e.g., `ommx_v1.SampleSet`). A `DataStore` can potentially hold multiple named `SampleSet` objects.
* **Content:** Objective values, solution status (feasible, optimal), variable assignments, timing information, energy values, sample occurrences, etc.
* **Logging:** Populated using `log_result()` (often for single solutions) or `log_sampleset()` (for collections of samples).

### 4. Metadata (`.metadata`)

* **Purpose:** Stores contextual information *about* the experiment itself, crucial for reproducibility and tracking.
* **Content:** This can include timestamps (start/end times), execution environment details (OS, library versions), Git commit hashes, user comments, unique run identifiers, or any other contextual data relevant to understanding when, where, and how the experiment was performed.
* **Format:** Typically a dictionary-like structure (key-value pairs).
* **Logging:** Populated using `log_metadata()`.

### 5. Arbitrary Objects (`.objects`)

* **Purpose:** Provides a flexible way to store other Python objects that don't fit neatly into the predefined categories.
* **Content:** This could be custom analysis objects, complex data structures, or anything else relevant to the experiment.
* **Caveats:** Use with caution. Ensure the stored objects are serializable if you intend to save and load the `DataStore`. Over-reliance on this might reduce the structured benefit of Minto.
* **Logging:** Populated using `log_object()`.

## Populating a `DataStore`: The `log_xxx` Interface

It's important to remember that you typically don't modify the attributes (`.instance`, `.parameters`, etc.) directly. Instead, you use the suite of `log_xxx` functions provided by Minto. Each `log_xxx` function is designed to place the provided data into the correct category within the `DataStore`, ensuring consistency. See the [Logging Guide](../logging_guide.md) for details.

## `DataStore` as a Unit of Information

As mentioned in the overview, a single `DataStore` object serves as the unit of information. This unit can represent:

1.  **Experiment-Wide Data:** A `DataStore` holding information common to many runs (e.g., the problem instance).
2.  **Run-Specific Data:** A `DataStore` holding information unique to one specific run (e.g., parameters and results for that run).

Understanding this dual role is key to using `DataStore` effectively within the broader `ExperimentDataSpace` structure for efficient data organization.

## Persistence: Saving and Loading

A `DataStore` object, along with all the structured data it contains, can be easily saved to storage (e.g., your file system) using the `.save()` method and loaded back later using `minto.load()`. This allows you to persist your experimental results and reload them for analysis without rerunning the experiment.

## Summary: Key Takeaways

* `DataStore` is the central container for **all data related to one experimental context** (run or experiment-level).
* Think of it as a **structured digital notebook** with predefined categories: `instance`, `parameters`, `results` (`solution`/`sampleset`), `metadata`, and `objects`.
* It's populated using the **`log_xxx` functions**.
* It serves as the fundamental **unit of information** managed by `ExperimentDataSpace`.
* It can be easily **saved and loaded**.

## Next Steps

Now that you have a better understanding of the `DataStore`'s structure and purpose, explore how it fits into the larger picture:

* Understand how multiple `DataStore`s (representing runs and common data) are organized in the [Mental Model](../mental_model.md) guide.
* [Logging Guide](../logging_guide.md): See practical examples of how to use the `log_xxx` functions to populate a `DataStore`.