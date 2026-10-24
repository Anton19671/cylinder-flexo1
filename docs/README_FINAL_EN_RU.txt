CYLINDER SOLVER — FULL USER GUIDE
=================================
Closed circular cylindrical shell solver

ENGLISH — COMPLETE OPERATING GUIDE
==================================

1. WINDOWS STARTUP
------------------
1. Fully extract the ZIP archive into a normal folder.
2. Run START_PROGRAM.bat (recommended) or START_PROGRAM.cmd.
3. The startup sequence checks Python, tkinter, pip, NumPy and Matplotlib.
4. Missing NumPy/Matplotlib packages may be installed with pip --user.
5. The GUI is launched in a separate pythonw.exe process; the startup console may then be closed.
6. Runtime GUI errors are written to GUI_RUNTIME_LOG.txt.
7. Do not run the program directly from inside a ZIP viewer.

2. IMPLEMENTED THEORIES AND METHODS
-----------------------------------
- Carrera Unified Formulation (CUF) and Kirchhoff-Love shell theory.
- CUF Taylor expansions TE1, TE2, TE3 and TE4.
- Variational Iteration Method (MVI) and independent strong-form FDM/MKR route.
- MVI ranks 1...8 and a common MVI-1...8 series launcher.
- Linear elastic, linear Birger, nonlinear elastic and nonlinear Birger modes where supported.
- Explicit end boundary-condition presets plus an advanced individual boundary-condition editor.
- FREE-FREE is rejected unless rigid modes are handled explicitly.

3. UNITS
--------
The GUI uses one explicit consistent unit system:
- N_MM_MPA: lengths/displacements in mm, E and stresses in MPa, pressure in N/mm^2, line load in N/mm.
- SI_N_M_PA: lengths/displacements in m, E and stresses in Pa, pressure in Pa, line load in N/m.
The selected unit system is stored in CASE.units.system. The GUI converts its own dimensional input fields when the unit system changes. Arbitrary external NPZ/CSV/formula data are never silently rescaled. Suspicious unit magnitudes produce warnings rather than hidden corrections. The mechanics kernel works in the declared consistent system.

4. GRID AND THICKNESS QUADRATURE
--------------------------------
MVI and FDM/MKR meshes are configured independently. nx and ns belong to the active in-plane discretization. nz is the number of Gauss points through the thickness, not a third spatial mesh. Layered quadrature uses physical layer boundaries and Gauss points per layer. The GUI reports Delta x, Delta s and the number of full FDM unknowns.

5. LOADS
---------
Supported radial loading includes uniform face traction, physical patch traction, a line-load generator using a finite physical band width, external NPZ generalized loads and harmonic radial loading where enabled. CUF allows OUTER/INNER face selection. Load factors are explicit. Current surface/generalized loads use DEAD_REFERENCE kinematics; follower-current-normal loading is deliberately rejected because it requires a deformation-dependent normal/area and a consistent tangent.

6. BIRGER MATERIAL LAW
----------------------
Phi(epsilon_i) can be defined by TABLE, strict CSV/TXT, FORMULA, PIECEWISE or EXTERNAL_PYTHON. TABLE/CSV extrapolation is explicit: ERROR, LINEAR or HOLD. No hidden spline or automatic polynomial fitting is used. The curve is an intensity relation sigma_i = Phi(epsilon_i), not an ordinary uniaxial stress-strain curve. The initial slope must be compatible with 3*G0 and the curve must start from (0,0). K5/K6 checks verify the origin, initial slope and immediate secant-GB continuity. User-defined yield stress/strain is used for visualization of plastic/nonlinear zones.

For finite-deformation CUF Green-Lagrange branches, the work-conjugate pair is Green-Lagrange strain E and second Piola-Kirchhoff stress S. The Birger law is therefore interpreted in that constitutive setting; direct identification with a small-strain experimental intensity curve at large strain requires separate physical justification.

7. MVI
------
Default MVI X/Y profile banks are generated automatically and stored in RUN/INPUT/AUTO_INITIAL_PROFILES_MVI.npz. An expert may provide an external NPZ profile bank. CUF profile dimensions follow the chosen TE order. The production MVI route does not introduce hidden SVD/pinv/lstsq/QR/Cholesky fallbacks, hidden regularization or hidden full-grid correction. Any explicit gauge/balancing operation used by a particular verified branch is documented by that branch.

8. STRONG-FORM FDM / NEWTON
----------------------------
The strong collocation matrix is generally nonsymmetric; this is expected and the solver uses a general linear solve. Linear CUF uses the strong FDM system. Nonlinear CUF keeps the moderate von Karman branch separate from the experimental full Green-Lagrange branch. Nonlinear tolerances and Jacobian settings are explicit. Where exact analytic tangents are implemented they are used directly; verified finite-difference Jacobians remain reference/regression paths. detF guards prevent accepting invalid finite-deformation states.

9. OUTPUT AND POSTPROCESSING
----------------------------
Postprocessing is started manually by the user after a calculation. The GUI provides three independent buttons: FAST / БЫСТРЫЙ, STANDARD / СТАНДАРТНЫЙ and FULL / ПОЛНЫЙ. No postprocessing mode starts automatically after RUN or STOP. FAST creates a compact engineering-control output; STANDARD creates the normal engineering package with key 2D/3D fields, sections and Excel data; FULL creates the complete available field, section, VTK and extended Excel output. A later mode can be started from the already saved RUN without solving the mechanics problem again.

The 3D engineering output includes a bilingual 3x3 cylinder overview: deformed shape, von Mises stress intensity and von Mises strain intensity, each shown in isometric, side and end views. All three views in one row use exactly the same physical colour normalization derived from the calculated array. Display smoothing affects visualization only and does not change numerical extrema or exported field values.

The principal load-deflection curve uses a fixed physical control point rather than following the location of the current global maximum. Diagnostic |w|max is exported separately. Birger runs additionally export GB/G0, intensity measures and nonlinear/plastic-zone information when the required criteria are defined.

Each future RUN README is bilingual: README_RESULTS_EN_RU.txt contains the English section first and the Russian section second. A legacy README_RESULTS_RU.txt alias is retained only for compatibility and contains the same bilingual text.

10. RESULTS DIRECTORY
---------------------
Runs are stored as RUN_001, RUN_002, ... without overwriting earlier runs. RUNS_INDEX.csv is the run index. Exact run parameters are preserved in INPUT, RUN_INFO.json and PARAMETERS.txt. The result tree separates Excel, CSV, 2D graphs, 3D results, VTK, machine-readable numerical data and reports.

11. STOP / PARTIAL RESULTS
--------------------------
Stopping the calculation does not delete previously accepted steps. Postprocessing may be performed from the last accepted state. An unaccepted trial state is never presented as a converged result. Failure diagnostics distinguish invalid deformation mapping, MVI half-step nonconvergence and other explicit stop reasons.

12. VERIFICATION AND SELF-CHECKS
--------------------------------
The package contains regression/self-check scripts for the principal CUF/KL, MVI/FDM, unit-system, thickness quadrature, Birger, boundary-condition, tangent/Jacobian and postprocessing paths. These are software/integration checks unless a case is explicitly identified as a physical benchmark. Run FINAL_SELF_CHECK.py or RUN_SELF_CHECK.cmd for the main quick regression suite.

13. DEPENDENCIES
----------------
Python 3, NumPy and Matplotlib are required. tkinter must be included in the Python installation. ParaView is optional and is used only to inspect exported VTK files.

14. IMPORTANT IMPLEMENTATION NOTES
----------------------------------
- The solver keeps linear, von Karman and full Green-Lagrange kinematics as distinct branches.
- The full Green-Lagrange branch uses detF > 0 validity checks.
- The Birger curve is not silently smoothed or repaired. Descending branches are rejected where the active validation rules require a nondecreasing intensity law.
- Through-thickness fields use the selected Gauss/layered quadrature and preserve quadrature metadata when state compatibility matters.
- Plastic-volume measures use the actual quadrature weights and the cylindrical metric factor where implemented.
- Load kinematics, unit system, theory, material model, method, boundary conditions, grid and postprocessing settings are written to the run data so that the case can be reproduced.

The Russian section below retains the detailed implementation history and branch-specific notes supplied with the solver.

РУССКИЙ — ПОЛНОЕ РУКОВОДСТВО
============================

РУЧНАЯ ПОСТОБРАБОТКА — FAST / STANDARD / FULL
------------------------------------------------
Постобработка запускается только пользователем. После расчёта доступны три независимые кнопки: FAST / БЫСТРЫЙ, STANDARD / СТАНДАРТНЫЙ и FULL / ПОЛНЫЙ. После RUN или STOP ни один режим автоматически не запускается. FAST формирует лёгкий контрольный набор; STANDARD — основной инженерный набор с ключевыми 2D/3D полями, сечениями и Excel; FULL — полный доступный набор полей, сечений, VTK и расширенных Excel. Повторно решать задачу для запуска другого режима не требуется: постпроцессор использует сохранённый RUN.

Для 3D формируется единый двуязычный лист 3x3: деформированная форма, интенсивность напряжений по Мизесу и интенсивность деформаций по Мизесу; для каждого поля — аксонометрия, боковой вид и вид на торец. В пределах одной строки используется одна физическая цветовая шкала, вычисленная по расчётному массиву. Сглаживание применяется только к визуализации и не изменяет численные значения и экстремумы.


CYLINDER SOLVER — STAGE 12 FULL INTERFACE
Замкнутая круговая цилиндрическая оболочка

ЗАПУСК В WINDOWS
1. Полностью распаковать ZIP в обычную папку.
2. Запустить START_PROGRAM.cmd.
3. START_PROGRAM сам проверяет Python, tkinter, pip, NumPy и Matplotlib.
4. Если NumPy/Matplotlib отсутствуют, START_PROGRAM пытается установить их через pip --user.
5. После успешной проверки START_PROGRAM запускает GUI через отдельный pythonw.exe-процесс. Чёрное окно запуска можно закрывать: интерфейс продолжит работать независимо.
6. Ошибки уже запущенного GUI записываются в GUI_RUNTIME_LOG.txt.
6. Не запускать программу прямо из окна ZIP-архиватора.

ЧТО ПОДКЛЮЧЕНО
- Теории: CUF и Kirchhoff–Love.
- CUF Taylor Expansion: TE1, TE2, TE3, TE4.
  TE1: F={1,z}; TE2: {1,z,z^2}; TE3: {1,z,z^2,z^3}; TE4: {1,z,z^2,z^3,z^4}.
- Методы: МВИ и независимый strong FDM/МКР.
- МВИ-N: N=1…8; отдельный запуск серии МВИ-1…8.
- Режимы: LINEAR_ELASTIC, LINEAR_BIRGER, NONLINEAR_ELASTIC, NONLINEAR_BIRGER.
- Простые C/F торцевые КУ, четыре явно подписанных варианта шарнира и
  индивидуальный редактор:
  S1: Nx=0, v=w=0, Mx=0; S2: u=w=0, Nxs=Mx=0;
  S3: u=v=w=0, Mx=0; S4: Nx=Nxs=0, w=0, Mx=0.
  Для KL S1-S4 задаются через u/v/w/slope. Для CUF S1-S4 задаются как
  явные обобщённые пресеты GEOMETRIC/NATURAL по всем u_tau/v_tau/w_tau
  выбранной TE-модели и могут быть раскрыты в индивидуальном редакторе.
  Для CUF индивидуальный редактор автоматически показывает все u_tau/v_tau/w_tau выбранной TE-модели.
- FREE–FREE без отдельной обработки жёстких мод явно отклоняется.

ЕДИНИЦЫ — V5
В GUI явно выбирается одна согласованная система:
- N_MM_MPA: длины/перемещения mm, E и напряжения MPa, давление N/mm^2, line load N/mm.
- SI_N_M_PA: длины/перемещения m, E и напряжения Pa, давление Pa, line load N/m.
Поле CASE.units.system обязательно: неявной системы единиц больше нет. При смене системы в GUI собственные числовые поля (геометрия, E0/GB/sigma, встроенные нагрузки, TABLE-кривая Биргера и x контрольных точек) пересчитываются численно, поэтому физическая задача не меняется только из-за смены подписи. Если подключены внешние NPZ, расширенные КУ или произвольная FORMULA/CSV/EXTERNAL_PYTHON-кривая, автоматическая смена блокируется: размерность произвольных внешних данных решатель не угадывает.
Перед расчётом диспетчер интерпретирует размерные скаляры в объявленной системе и переводит их в SI для диагностического контроля. Очевидно подозрительные масштабы mm/m или MPa/Pa дают явное V5-предупреждение, но НЕ блокируют нестандартный материал: например, E0=2.1e11 при units.system=N_MM_MPA помечается как нетипичное для MPa и предлагается проверить систему единиц. Механическое ядро по-прежнему работает в согласованной системе и не выполняет скрытых преобразований.

СЕТКА
- Сетки МВИ и МКР/FDM задаются раздельно: у каждого метода собственные nx и ns. В обычном режиме показывается сетка активного метода, в экспертном доступны обе.
- Поле `nz` в GUI означает число точек Гаусса по толщине. Для оболочечных KL/CUF это квадратура по толщине, а не третья пространственная сетка. Для LAYERED задаются физические z-границы слоёв и число Gauss-точек на слой.
- Вкладка «Постановка» имеет вертикальную прокрутку, поэтому сетка и начальные профили МВИ доступны при любом масштабе экрана Windows.
- GUI показывает Delta x, Delta s и число неизвестных полного FDM-поля.
- Пресет 21×21 / 41×41 / 61×61 записывает nx и ns именно в активную сетку, включая переключение между МВИ и МКР; это удобный явный ввод, а не скрытая настройка.

НАГРУЗКА
Встроенный проверенный генератор поддерживает радиальные face traction:
- UNIFORM_RADIAL_FACE;
- PATCH_RADIAL_FACE по фиксированным физическим границам x1..x2 и theta1..theta2 либо s1..s2; внутренний фронт сглаживается C1-переходом cos^2 на ширину половины активного FD-шаблона (для 7-точечного шаблона — 3 шага сетки).
- LINE_RADIAL_GENERATOR — линейная интенсивность p_line задаётся через полосу ФИЗИЧЕСКОЙ ширины band_width, не зависящей от сетки; требуется band_width >= 2*Delta s. Старое q=p_line/Delta s в одном узле запрещено как несходящееся в сильной коллокационной постановке;
- EXTERNAL_NPZ для произвольного заранее подготовленного generalized-load массива.
Для CUF явно выбирается OUTER/INNER face. Нагрузка может иметь несколько явных load_factors.
V8: вся текущая поверхностная/generalized нагрузка имеет кинематику DEAD_REFERENCE — это номинальная (мёртвая) нагрузка, фиксированная в отсчётной конфигурации. Направление и величина не следуют за деформацией. FOLLOWER_CURRENT_NORMAL намеренно отклоняется как нереализованная постановка: полноценная следящая нагрузка требует деформационно-зависимых нормали/площади и собственного consistent tangent. Поле load_kinematics сохраняется в расчётном отчёте.
GUI содержит предпросмотр карты q(x,theta) и схему цилиндра.

КРИВАЯ БИРГЕРА
Phi(epsilon_i) задаётся непосредственно из GUI одним из способов:
- TABLE: пары epsilon_i, sigma_i;
- CSV/TXT: V9-строгий разбор двух колонок; поддерживаются разделители запятая, точка с запятой и TAB, а для разделителя «;» поддерживается десятичная запятая. Допустим только один начальный заголовок; испорченная строка данных приводит к явной ошибке с номером строки и никогда не пропускается молча;
- FORMULA;
- PIECEWISE;
- EXTERNAL_PYTHON.
Экстраполяция для TABLE/CSV выбирается явно: ERROR / LINEAR / HOLD.
Автоматической полиномиальной подгонки или скрытого spline нет.
ВАЖНО: это диаграмма интенсивностей sigma_i = Phi(epsilon_i), а не обычная одноосная sigma-epsilon.

ПОСТАНОВКА БИРГЕРА ПРИ ГЕОМЕТРИЧЕСКОЙ НЕЛИНЕЙНОСТИ
В конечнодеформационной CUF-ветке используется энергетически сопряжённая пара:
E — тензор деформаций Green-Lagrange, S — второй тензор напряжений Piola-Kirchhoff.
Материальный закон записан как S = K0*tr(E)*I + 2*GB(epsilon_i)*dev(E), а epsilon_i и sigma_i
являются инвариантами именно этих тензоров. При малых деформациях эта постановка переходит
в классический метод Биргера переменных параметров упругости. При конечных деформациях это
St.Venant-Kirchhoff-type обобщение с переменным секущим модулем сдвига GB. Поэтому прямое
отождествление Phi(epsilon_i) с экспериментальной малодеформационной диаграммой интенсивностей
при больших деформациях требует отдельного физического обоснования. Проверка K5 контролирует
только начальный наклон при epsilon_i -> 0 и не является таким обоснованием.

Для совместимости с E0, nu0 начальный упругий наклон должен удовлетворять dPhi/depsilon_i = 3*G0,
где G0 = E0/[2(1+nu0)]. Кривая должна быть определена от epsilon_i=0 и проходить через (0,0).
Перед запуском выполняются K5/K6-проверки: начальный наклон (допуск initial_slope_rtol, по умолчанию 0.05),
нулевой интерсепт и отсутствие скачка секущего GB сразу выше epsilon_tol.
При необходимости пользователь задаёт yield_stress или yield_strain только для визуализации пластической зоны.

МВИ
- В обычном режиме начальные X/Y профили создаются автоматически и сохраняются внутри RUN/INPUT/AUTO_INITIAL_PROFILES_MVI.npz. Масштаб начального поля пропорционален толщине h, поэтому физически одинаков при системах мм-Н-МПа и м-Н-Па.
- Эксперт может явно выбрать собственный NPZ; при его наличии внешний файл имеет приоритет над автоматическим профилем.
- CUF profile-bank shapes:
  TE1 m=2, TE2 m=3, TE3 m=4, TE4 m=5;
  X=(3,m,rank,nx), Y=(3,m,rank,ns).
- Никаких SVD/pinv/lstsq/QR/Cholesky fallback, hidden regularization, relaxation или скрытой full-grid коррекции МВИ.

СВОЙСТВА ДИСКРЕТНОГО ОПЕРАТОРА — Z3
Сильная коллокационная постановка даёт в общем случае НЕСИММЕТРИЧНУЮ матрицу даже для линейной упругой задачи. Это ожидаемое свойство strong-form discretization, а не дефект механики. Поэтому нельзя автоматически применять разложения/методы, требующие симметрии (Cholesky, LDL^T, CG), нельзя трактовать знаки собственных значений как для симметричной энергетической касательной матрицы, а потерю устойчивости следует связывать с вырождением соответствующего оператора. Текущий решатель использует общий np.linalg.solve.

FDM / NEWTON
- Linear CUF решается сильной FDM-системой.
- Нелинейный CUF имеет две явно разделённые ветви: основную умеренно-нелинейную
  von Karman + Hooke/Birger и экспериментальную полную Green–Lagrange +
  Hooke S–E (StVK)/Birger S–E. Выбор одной ветви не подменяет другую.
- Newton tolerances, относительный FD Jacobian step (V3) и max iterations задаются пользователем.
  После V3 fd_step безразмерный: h_j = fd_step*max(|x_j|, characteristic_scale_j);
  используется центральная разность. Для CUF TE1 characteristic_scale: h для u0/v0/w0
  и h/L для u1/v1/w1. Для KL: h для u/v/w.
- Нет автоматического retry/шага нагрузки, fallback-решателей, SVD/pinv/lstsq/QR
  или нормировки X/Y. Для nonlinear CUF-MVI включён явный detF-guard:
  если полная MVI-поправка даёт пробное detF<=0 или увеличивает Rproj_rel,
  длина этой correction половинится до допустимого пробного состояния; если
  центральное FD-возмущение колонки касательной даёт detF<=0, уменьшается
  FD-step только этой колонки. События пишутся в LIVE_PROGRESS как
  mvi_correction_damped и mvi_tangent_damped.
  Default max_halfstep_iterations для новых GUI-запусков MVI установлен 80,
  потому что guarded correction может требовать больше внутренних итераций.

АВТОМАТИЧЕСКИЙ ВЫХОД
После успешного расчёта FINAL_POSTPROCESS создаёт:
- NUMERIC/ — derived NPZ/CSV;
- GRAPHS/DISPLACEMENTS;
- GRAPHS/STRAINS;
- GRAPHS/STRESSES;
- GRAPHS/RESULTANTS;
- GRAPHS/LOAD_DEFLECTION;
- GRAPHS/THROUGH_THICKNESS;
- GRAPHS/CONVERGENCE;
- GRAPHS/BIRGER только для Birger-режимов;
- 3D/DEFORMED_SHELL;
- 3D/LOAD_AND_BC;
- 3D/STRESS_FIELDS;
- 3D/BIRGER_ZONES только для Birger-режимов;
- REPORT/CALCULATION_SUMMARY.json;
- NUMERIC_RESULTS.zip.
Основной график нагрузки-прогиба: GRAPHS/LOAD_DEFLECTION/LOAD_DEFLECTION.png,
он строится по фиксированной точке в середине оболочки |w(center)|.
GRAPHS/LOAD_DEFLECTION/LOAD_WMAX_DIAGNOSTIC.png — диагностический максимум по
полю |w|max; он не является основным графиком, потому что точка максимума может
смениться по шагам.

Для Birger дополнительно:
- GB/G0 min/max;
- 3D GB/G0;
- sigma_i, epsilon_i;
- 3D plastic/nonlinear zone при явно заданном yield_stress/yield_strain;
- пользовательские пороги GB/G0 как зоны физической нелинейности.

GUI — ВКЛАДКА «РЕЗУЛЬТАТЫ»
- читает итоговый REPORT;
- показывает status, модель, метод, режим, число шагов и |w|max;
- находит PNG из GRAPHS и 3D;
- выбранный PNG отображается прямо внутри программы.

ПРОВЕРКИ STAGE 12
- Старый FINAL_SELF_CHECK: CUF-TE1 и KL — PASS, старые контрольные результаты не изменены.
- TE2/TE3/TE4 FDM: ненулевой LINEAR_ELASTIC software-regression — CONVERGED.
- TE2/TE3/TE4 MVI: zero-route smoke — CONVERGED; TE2/TE3/TE4 MVI-1 ненулевые software-smoke — CONVERGED с ненулевыми полями.
- TE2 LINEAR_BIRGER с TABLE-кривой — CONVERGED_BIRGER; postprocess Birger — PASS.
- TE4 postprocess: derived fields, generalized resultants, 3D stress/strain surfaces — PASS.
Эти tests являются software/integration checks, а не опубликованными физическими benchmark-задачами.

V2 — КРИТЕРИЙ NEWTON ПО НЕВЯЗКЕ
Абсолютный residual tolerance больше не используется в nonlinear FDM. Поле CASE называется residual_tolerance_relative и является безразмерным. Масштаб вычисляется один раз до Newton: max(||generalized_load||inf, ||R_initial||inf, 1). Рекомендуемое значение по умолчанию — 1e-9. Старое поле newton.residual_tolerance намеренно отклоняется, чтобы абсолютный допуск не был молча истолкован как относительный. Абсолютная residual_inf сохраняется в результатах вместе с residual_relative и residual_scale.

V5 — ПРОВЕРКА ЕДИНИЦ
Быстрый unit-regression: python V5_SELF_CHECK.py. Он проверяет точные преобразования N_MM_MPA <-> SI_N_M_PA, совпадение SI-интерпретации двух физически эквивалентных CASE и обязательное предупреждение на подозрительном E0=2.1e11, объявленном как MPa; предупреждение не запрещает запуск.

V6 — КВАДРАТУРА ПО ТОЛЩИНЕ
Доступны GAUSS и LAYERED. Существующая послойная квадратура подключена к FDM, MVI, GUI и постпроцессору. Для Биргера GAUSS с n_gauss<7 даёт предупреждение о возможном недостаточном разрешении упругопластической границы, но не блокируется. Быстрая проверка: python V6_SELF_CHECK.py.

V7 — ДОЛЯ ПЛАСТИЧЕСКОГО ОБЪЁМА
В отчёте plastic_volume_fraction считается с весами выбранной квадратуры и цилиндрическим множителем H=1+z/R. Старое невзвешенное среднее по точкам сохранено только как диагностическое plastic_point_fraction_unweighted; неоднозначный ключ plastic_volume_fraction_grid удалён. Быстрая проверка: python V7_SELF_CHECK.py.

V8 — КИНЕМАТИКА НАГРУЗКИ
Текущая проверенная постановка явно маркируется DEAD_REFERENCE. Следящая нагрузка не имитируется и не подменяется фиксированным RHS: запрос follower/current-normal явно отклоняется. Быстрая проверка: python V8_SELF_CHECK.py.

Z1 — МАСШТАБИРОВАННЫЕ FD-ВЕСА
Веса конечных разностей строятся в безразмерных локальных координатах stencil, после чего возвращается физический множитель h_c^{-order}. Это убирает искусственный рост cond(Вандермонда) при смене единиц/шага сетки без изменения самой FD-формулы. Быстрая проверка: python Z1_SELF_CHECK.py.

Z2 — СТАТИЧЕСКАЯ ГИГИЕНА КОДА
Удалены доказанно неиспользуемые импорты и мёртвые локальные присваивания в production-модулях. Специально проверена связь FULL Green–Lagrange + Birger: импорт fg НЕ удалён, потому что реально используется аналитической сборкой tangent. В release включён независимый stdlib-AST guard Z2_SELF_CHECK.py; внешняя зависимость pyflakes для запуска решателя не требуется.

Z4 — СОВМЕСТИМОСТЬ СОСТОЯНИЯ GB
GB хранится в точках толщинной квадратуры. STEP_STATE для Birger теперь сохраняет quadrature_z, quadrature_w и thickness_h. При initial_GB.source=NPZ решатель требует эти метаданные и сравнивает их с текущей квадратурой; legacy GB без метаданных или GB от другой h/квадратуры явно отклоняется, без скрытой интерполяции. Быстрая проверка: python Z4_SELF_CHECK.py.

Z5 — ЧИСТКА ВАЛИДАТОРА
Удалены неиспользуемые expected_tail_X/expected_tail_Y; фактическая проверка формы profile bank сохранена без изменения.

V9 — СТРОГИЙ CSV БИРГЕРА
CSV-кривая больше не теряет испорченные строки молча. Один начальный заголовок и комментарии разрешены; ошибка в данных сообщает номер физической строки. Разделитель определяется среди ',', ';' и TAB. Быстрая проверка: python V9_SELF_CHECK.py.

БЫСТРАЯ ПРОВЕРКА
python FINAL_SELF_CHECK.py
или RUN_SELF_CHECK.cmd

ЗАВИСИМОСТИ
Python 3, NumPy, Matplotlib; tkinter должен входить в установленный Python.
=== STAGE12 INTERFACE FIX 2026-08-14 ===

ИНТЕРФЕЙС:
- По умолчанию открывается обычный пользовательский режим: постановка, нагрузка, онлайн-расчёт и результаты. Служебные допуски, FD-step, лимиты циклов, ручные CASE/NPZ-пути, параметры постобработки и журнал вынесены в отдельный экспертный режим. Для МВИ ручной NPZ не требуется: начальные профили формируются автоматически.
- В верхней строке независимо и явно выбираются теория, геометрическая модель
  и материал. Для Carrera CUF доступны линейная, von Karman и полная
  Green–Lagrange; для Kirchhoff–Love — линейная и von Karman.
- Служебные параметры обычного режима получают проверенные значения по умолчанию и всё равно сохраняются в FINAL_CASE.json; эксперт может просмотреть и изменить их.
- Корневая папка по умолчанию — RESULTS на один уровень выше папки C с программой. Пользователь задаёт только название расчёта; каждый запуск автоматически создаёт новую короткую папку RUN без перезаписи.
- Сетка FDM/МКР и сетка МВИ задаются НЕЗАВИСИМО.
  В обычном режиме обе сетки видны одновременно и имеют понятные пресеты 21x21, 41x41, 61x61; после применения пресета их можно менять независимо. Рядом отдельно задано nz — число точек Гаусса по толщине.
  Раздельные FD stencil доступны в экспертном режиме.
  В CASE сохраняются одновременно fdm_grid и mvi_grid; grid остаётся совместимым активным представлением.
- Нагрузки: UNIFORM_RADIAL_FACE, PATCH_RADIAL_FACE, LINE_RADIAL_GENERATOR, EXTERNAL_NPZ.
- Для PATCH задаются x1, x2 и окружные координаты c1, c2 (THETA или S); можно выбрать всю окружность. Размер площадки остаётся физически тем же при изменении nx/ns; сглаживание фронта выполняется автоматически.
- Для LINE задаются положение c0, p_line и обязательная физическая ширина полосы band_width.
- Шаг нагружения задаётся явно: COUNT (lambda_end и число шагов), RANGE
  (lambda_start, lambda_step, lambda_end) или MANUAL_LIST. В обычном режиме
  COUNT по умолчанию: lambda_end=1.0 и 20 шагов; фактическая нагрузка шага
  равна lambda*q. Поле p_line показывается только для LINE_RADIAL_GENERATOR
  и скрыто для равномерного давления.
- Материал: справочный generic isotropic список или USER с ручным E0, nu0. Справочник не заменяет проверку конкретной марки материала.
- Пользовательский материал можно сохранить под собственным названием кнопкой «Добавить в библиотеку». В MATERIAL_LIBRARY.json записываются название, E0 в Па и nu0; существующее имя заменяется только после подтверждения.

КРИВАЯ БИРГЕРА:
- TABLE: точки epsilon_i, sigma_i.
- CSV: импорт экспериментальных точек.
- FORMULA: выражение Phi(eps) и именованные численные параметры A=..., B=..., E=... .
- PIECEWISE: кусочно-заданная формула.
- EXTERNAL_PYTHON: сохранён для совместимости.
- График Phi(eps) встроен непосредственно во вкладку «Биргер».
- Импорт/экспорт всей кривой: BIRGER_CURVE.json.
- Импорт/экспорт дискретной кривой: CSV.
- Экспорт графика: PNG.
- sigma_T / epsilon_T используются только как явно заданные пользователем критерии визуализации пластической зоны.
- K5/K6: Phi(0)=0; начальный наклон Phi'(0)=3G0; обычная одноосная sigma-epsilon без преобразования отклоняется.

СОХРАНЕНИЕ:
- «Сохранить CASE / Загрузить CASE» — валидируемый FINAL_CASE.json для расчёта.
- «Открыть параметры / Сохранить / Сохранить как…» — CYLINDER_GUI_PARAMETERS.json.
  Этот файл сохраняет даже НЕЗАВЕРШЁННЫЙ ввод интерфейса и затем полностью восстанавливает поля GUI.

ПРОВЕРКИ ПОСЛЕ ПРАВОК:
- FINAL_SELF_CHECK: PASS; CUF_TE1 difference=0.0; KL difference=0.0.
- Независимые FDM 9x12 и MVI 15x24 сетки сохраняются и восстанавливаются раздельно.
- FORMULA A*eps+B*eps**2 с параметрами A/B: контрольная ошибка 0.0.
- PATCH + RANGE: координаты и факторы нагрузки сохраняются/восстанавливаются.
- Software-only TE1 FDM PATCH run без loads_npz: 2/2 шагов сошлись; residual_inf 3.6193270602780103e-14 и 7.238654120556021e-14.
- Solver mechanics при этой интерфейсной правке не переписывались.


ВЕРХНЯЯ ПАНЕЛЬ УПРАВЛЕНИЯ
--------------------------
На постоянной верхней панели доступны:
- Открыть параметры
- Сохранить параметры
- Сохранить как...
- Пуск
- Стоп
- Открыть папку результатов
Кнопка Стоп завершает только текущий расчётный subprocess и не закрывает GUI.


STAGE12 PREFLIGHT FIX
- FDM/MVI stencil is explicit readonly 7-point verified scheme.
- Start/Validate performs a single preflight check and lists all missing mandatory fields before any subprocess starts.
- Old saved GUI parameter files with empty stencil are restored with stencil=7.
- Solver mechanics unchanged.

=== ДОПОЛНЕНИЕ K2 + V10 ===

V10:
- TABLE/CSV: заданные точки Phi(epsilon_i) обязаны быть неубывающими.
- FORMULA/PIECEWISE/EXTERNAL_PYTHON: выполняется численная проверка на 0 <= epsilon_i <= epsilon_i_limit.
- Упрочнение и плато разрешены; нисходящая ветвь отклоняется. Кривая не сглаживается и не исправляется автоматически.

K2:
- Применён к полной CUF FDM-ветке: LINEAR_ELASTIC, NONLINEAR_ELASTIC,
  LINEAR_BIRGER (frozen GB), NONLINEAR_BIRGER (frozen GB Jacobian).
- Итоговая A/J остаётся плотной numpy-матрицей и решается прежним np.linalg.solve
  через существующее точное диагональное масштабирование.
- Раскраска НЕ использует небезопасное is mod P. Области влияния строятся по
  фактическим ненулевым шаблонам D1x/D1s с двухшаговым замыканием
  q -> qx/qs/resultants -> strong residual. Периодический шов s учитывается
  непосредственно матрицей D1s; односторонние x-шаблоны учитываются D1x.
- Старые поодиночные dense-сборщики оставлены только как private reference для K2_SELF_CHECK.py.
- K2_SELF_CHECK.py сравнивает каждый элемент A/J, r0/RHS, решение q и wmax.
- CUF_FOUR_MODE_SELF_CHECK.py прогоняет четыре CUF-TE1 режима.


=== ДОПОЛНЕНИЕ K2b: ВЕКТОРИЗАЦИЯ ТОЛЩИННОГО ЦИКЛА ===

- Производственные general_thickness_resultants и general_linear_thickness_resultants
  больше не обходят каждый поверхностный узел Python-циклом.
- Внешний Python-цикл оставлен только по точкам квадратуры по толщине; все узлы
  поверхности обрабатываются пакетными NumPy-операциями.
- Полная Green-Lagrange ветка сохраняет обязательную проверку detF > 0 ДО
  вычисления определяющих соотношений.
- Поддержаны те же три формы GB: скаляр, поле на поверхности, полное поле
  (n_gauss, ns, nx).
- Старые скалярные реализации сохранены только как *_scalar_reference для
  K2B_SELF_CHECK.py и не используются рабочим решателем.
- Векторизованы также material-point evaluators CUF Linear/Nonlinear для
  Birger-веток MVI и FDM.
- Никаких SciPy/sparse/SVD/pinv/lstsq/QR/регуляризации/line-search не добавлено.
- FINAL_SELF_CHECK включает отдельный K2b regression guard.


=== ДОПОЛНЕНИЕ V3A: МАСШТАБИРОВАННЫЙ ЦЕНТРАЛЬНЫЙ FD-JACOBIAN ===

- Реализована минимальная V3-правка из аудита для нелинейного полного FDM/МКР.
- Абсолютный одинаковый шаг по всем степеням свободы больше не используется.
- Для каждого столбца: typ_j=max(|x_j|, characteristic_scale_j),
  h_j=fd_step*typ_j, J[:,j]=(R(x+h_j e_j)-R(x-h_j e_j))/(2 h_j).
- fd_step в nonlinear FDM теперь БЕЗРАЗМЕРНЫЙ относительный шаг.
- CUF Taylor TE-N: characteristic_scale_tau = h/L^tau; в частности TE1:
  tau=0 -> h, tau=1 -> h/L, как указано в аудите.
- KL: u/v/w используют characteristic_scale=h.
- CUF сохраняет безопасную K2-раскраску: разные h_j внутри одного цвета допустимы,
  потому что столбцы группы имеют непересекающиеся реальные области влияния;
  каждый извлечённый столбец делится на собственное 2*h_j.
- Старые одностолбцовые CUF-сборщики превращены в V3 central reference и используются
  только для self-check; рабочий CUF-путь остаётся K2.
- Проверены nonlinear CUF/KL Elastic и frozen-GB Birger.
- V3_SELF_CHECK дополнительно сравнивает J*v с независимой 5-точечной направленной
  производной и проверяет эквивалентные mm/MPa и SI постановки.
- V3A central-FD сохранён как независимый reference/self-check путь.
- MVI tangent_fd_step не изменён: аудит V3 задаёт characteristic_scale для полного
  nodal FDM-вектора, но не задаёт корректную шкалу для редуцированных X/Y-профилей;
  придумывать её без вывода не стали.

=== ДОПОЛНЕНИЕ V3B: ТОЧНЫЙ АНАЛИТИЧЕСКИЙ ELASTIC TANGENT ===

- NONLINEAR_ELASTIC CUF FDM: рабочий Newton больше не строит Jacobian конечными разностями.
  Реализована точная цепочка dF -> dE=sym(F^T dF) -> dS -> dP=dF*S+F*dS ->
  d(resultants) -> d(strong residual) с точной линеаризацией GEOMETRIC/NATURAL end BC.
- CUF tangent общий для TE1..TE4; regression J*v выполнен для TE1/TE2/TE3.
- CUF dense J собирается из точных tangent-action и безопасной K2-раскраски по реальному
  графу D1x/D1s. На TE1 15x15 exact-colored и exact-dense матрицы совпадают bit-for-bit;
  630 tangent-action вместо 1350 отдельных столбцов.
- NONLINEAR_ELASTIC KL FDM: реализована точная линеаризация von Karman мембранных членов,
  изгибных членов и всех end quantities (Tu,Tv,Tw,Ttheta,theta_n).
- KL поддерживает оба elastic resultants_method: ANALYTIC и QUADRATURE.
- V3A scaled-central FD Jacobian не удалён: он остаётся независимым эталоном для regression.
- Production NONLINEAR_ELASTIC CUF/KL вызывает analytic_jacobian; self-check отдельно запрещает
  обращение к FD-Jacobian в этой ветке.
- Birger consistent tangent в V3B НЕ добавлялся: NONLINEAR_BIRGER по-прежнему использует
  frozen-GB FD Jacobian. Это следующий отдельный этап V3C, чтобы не смешивать elastic и
  constitutive derivative dGB/depsilon_i.
- V3B_SELF_CHECK проверяет analytic J*v против независимой 5-точечной производной,
  analytic J против V3A FD reference, Newton state/iterations и чистый production path.



V11.2: для сгенерированного PATCH нелинейный FDM отдельно выводит невязку interior / boundary / load-front. Фронт cos^2 не ограничивает остановку Newton только если площадка разрешена сеткой и существует ненулевая нагруженная strong-row вне фронта; иначе автоматически используется полный residual.


HARMONIC_RADIAL_FACE (добавлено для универсальных гармонических нагрузок):
q_r(x,s) = q0*cos(n*s/R)*sin(m*pi*x/L), m>=1, n>=0.
Пример Nwoji Case 2: m=1, n=4. Это общий тип нагрузки, не зашитый benchmark.

=== ONLINE / РАСШИРЕННЫЕ РЕЗУЛЬТАТЫ / ОТДЕЛЬНЫЕ EXCEL 2026-08-20 ===
- По умолчанию корень результатов находится в папке `RESULTS` на один уровень выше папки `C` с решателем. В экспертном режиме корень можно изменить. Структура: RESULTS/<ИМЯ РАСЧЁТА>/RUN_001__<КОРОТКАЯ_МЕТКА>/, RUN_002__<КОРОТКАЯ_МЕТКА>/... . Пример метки: `TE1_MVI1_NL_Q10000_g41x41`. Если имя расчёта не введено, оно автоматически строится коротко из теории, режима, КУ и нагрузки. Предыдущие RUN не перезаписываются.
- В Windows каталог результатов по умолчанию — C:\CYL_RESULTS. Папки RUN_001, RUN_002, ... создаются прямо в нём, без промежуточного CASE-каталога. Там же создаются README_RESULTS_EN_RU.txt (и совместимый alias README_RESULTS_RU.txt) и короткий общий индекс RUNS_INDEX.csv.
- Внутри каждого варианта автоматически сохраняются INPUT/FINAL_CASE.json, INPUT/RUN_PARAMETERS.json, INPUT/GUI_PRESET.json, RUN_INFO.json и PARAMETERS.txt. Полная длинная расшифровка варианта хранится в RUN_INFO.json, PARAMETERS.txt и RUNS_INDEX.csv, а не в имени папки. Автоматические профили МВИ сохраняются в INPUT/AUTO_INITIAL_PROFILES_MVI.npz. Внешние NPZ/CSV по возможности копируются в INPUT/FILES, поэтому вариант можно воспроизвести без повторного ручного ввода.
- В каждом RUN создаются понятные папки EXCEL, CSV, GRAPHS, 3D, NUMERIC, REPORT и INPUT, а также README_RESULTS_EN_RU.txt с разделами English/Русский (совместимый README_RESULTS_RU.txt содержит тот же текст).
- EXCEL содержит полную книгу *FULL_RESULTS.xlsx и отдельные книги LOAD_DEFLECTION, DISPLACEMENTS, STRAINS_STRESSES, RESULTANTS, CONVERGENCE, CONTROL_POINTS и CUF_COEFFICIENTS, когда ветка CUF активна. Для МВИ-1…8 каждый ранг получает собственный набор.
- CSV и все PNG получают префикс имени расчёта, номера RUN, метода и режима. Графики остаются разложены по тематическим подпапкам: перемещения, деформации, напряжения, результанты, сходимость, пластичность, Биргер, 3D.
- Excel содержит параметры, сводку шагов, нагрузка–прогиб от (0,0), online/Newton/MVI/Birger histories, surface fields, through-thickness fields, CUF coefficients, standard-z displacements и resultants — только когда соответствующие данные реально существуют.
- Добавлена отдельная вкладка «Онлайн»: elapsed time, шаг/λ, стадия, wmax; Newton residual/increment; MVI half-step/cycle/Rproj/field change; Birger outer/eta_G/eta_sigma/GB/G0; таблица завершённых шагов; нагрузка–прогиб; 3D принятого шага; поле/пластичность; графики сходимости; текстовый журнал с автоматической прокруткой. Для МВИ журнал показывает номер шага нагрузки, λ и q, отсутствие автоматического дробления, цикл X↔Y, полушаг X/Y, внутреннюю итерацию, Rproj и ход построения касательной «колонка/всего, %». Для KL дополнительно раздельно показаны безразмерный Rproj_rel (критерий остановки) и размерный Rproj_abs (диагностика). Ручная прокрутка вверх приостанавливает автоматический возврат к новой строке; при возврате вниз автопрокрутка включается снова, также доступна кнопка «Автопрокрутка: вкл/выкл». Кнопка «Открыть большой журнал» выводит тот же поток в отдельное растягиваемое и разворачиваемое окно с зелёной прокруткой. Полоса выполнения зелёная.
- Для каждого графика «Онлайн» (нагрузка–прогиб, 3D, поле, сходимость) и каждого выбранного графика результатов есть команда открытия в отдельном изменяемом окне. В нём доступна стандартная панель Matplotlib: масштабирование, перемещение, возврат вида и сохранение изображения.
- Оси 2D/3D-графиков, цветовые шкалы и заголовки столбцов Excel содержат физические размерности выбранной системы единиц. Для встроенных нагрузок график «нагрузка–прогиб» показывает фактическую нагрузку q или p_line; λ сохраняется отдельным безразмерным столбцом. Внешняя NPZ-нагрузка без известной скалярной амплитуды остаётся на оси λ [-].
- Тяжёлые 3D/карты не обязаны перерисовываться на каждой внутренней итерации; скалярные показатели обновляются чаще, состояние тяжёлой визуализации берётся из принятых шагов.
- После Stop уже принятые шаги не удаляются. Для них запускается постобработка, поэтому остаются DERIVED, графики и Excel; незавершённый шаг не выдаётся за принятый.
- Если серия остановлена решателем после одного или нескольких принятых шагов,
  это не считается потерей результатов. FINAL_RUN_SUMMARY.json, RUN_RESULT.json
  и SERIES_SUMMARY.json содержат failed_step_code, failed_step_diagnostic_ru и
  failed_step_recommendation_ru. Например INVALID_DEFORMATION_MAPPING означает,
  что следующий пробный шаг дал detF<=0 и его нельзя принимать; принятые шаги
  до остановки остаются последней допустимой веткой. MVI_HALFSTEP_NOT_CONVERGED
  означает, что внутренний MVI halfstep не достиг projected_tolerance за
  заданный лимит итераций. Автоматического скрытого дробления нагрузки,
  fallback-решателей или нормировки X/Y нет; detF-guard внутренней correction
  и FD-касательной выводится явно в LIVE_PROGRESS и SERIES_SUMMARY.
- Для CUF/Carrera на каждый принятый шаг сохраняются q={u_tau,v_tau,w_tau}, физические u/v/w по толщине, 6 деформаций, 6 напряжений, epsilon_i, sigma_i, detF, quadrature z/w, resultants, standard-z displacements и деформированная срединная поверхность.
- Для Birger сохраняются GB/G0, нелинейные зоны и взвешенная доля по толщине. При явном yield_stress/yield_strain дополнительно сохраняются plastic_mask(x,s,z), hp/h, Vp/V, нижний/средний/верхний слой, x-z и theta-z разрезы и координаты первого появления пластической зоны.
- Доли по толщине/объёму используют реальные веса квадратуры и цилиндрический множитель H=1+z/R.
- Детали и перечень проверок: RELEASE_NOTES_ONLINE_RESULTS_EXCEL.txt.
=== ДОПОЛНЕНИЕ v13: ЭКСПЕРИМЕНТАЛЬНЫЙ CUF-МВИ ПО АЛГОРИТМУ ПЛАСТИНКИ ===

В отдельной v13-ветке CUF-МВИ выполняет один Y-полушаг и один X-полушаг
за цикл, по одной редуцированной поправке на полушаг. Начальные X-профили
независимы, Y=0. Перед полушагом применяется точное QR gauge-преобразование
замороженного семейства, после поправки — балансировка X/Y с сохранением
физического поля. Внутренний цикл 1...80 для CUF в этой ветке не используется.
Уравнения CUF, Green-Lagrange кинематика, касательная, нагрузка и граничные
условия не менялись. KL-МВИ оставлен на алгоритме v11. Полное описание и
результаты быстрых проверок: RELEASE_NOTES_CUF_PLATE_STYLE_MVI_V13.txt.

=== ДОПОЛНЕНИЕ v14: ЯВНОЕ МЕНЮ КИНЕМАТИКИ И МАТЕРИАЛА ===

Основной режим по умолчанию:
  Carrera CUF TE1 + von Karman + закон Гука + МВИ-1.

Carrera CUF TE1--TE4, МВИ или МКР/FDM:
  1) Линейная + закон Гука;
  2) Линейная + Биргер;
  3) von Karman + закон Гука;
  4) von Karman + Биргер;
  5) Green--Lagrange + Гук S--E (StVK), экспериментальный режим;
  6) Green--Lagrange + Биргер S--E, экспериментальный режим.

Контрольная Kirchhoff--Love, МВИ или МКР/FDM:
  1) Линейная + закон Гука;
  2) Линейная + Биргер;
  3) von Karman + закон Гука;
  4) von Karman + Биргер.

В CUF-von-Karman сохраняются цилиндрическая метрика H=1+z/R и мембранный
член w/R. Квадратичные члены построены по радиальным наклонам w,x и w,s/H,
а закон Гука/Биргера применяется непосредственно к умеренно-нелинейной
деформации. Деформационный градиент полной Green--Lagrange в эту ветвь не
подмешивается.

Алгоритм МВИ v13, профильный банк, касательные полушагов, критерии сходимости,
общая история нагрузки МВИ-1--8 и независимый запуск рангов в v14 не менялись.
Полные расчёты в поставке v14 не запускались; включены только быстрые проверки
CUF_VK_MODE_SELF_CHECK_V14.py, PLATE_STYLE_MVI_SELF_CHECK_V13.py и
MVI_1_8_COMMON_LOAD_SELF_CHECK.py.

=== ДОПОЛНЕНИЕ v15: СОГЛАСОВАННЫЙ D2 В KL-МВИ ===

Исправлена подтверждённая ошибка: KL-кинематика использовала прямые D2x/D2s,
а вариации проекционного остатка МВИ — отличающиеся D1x@D1x/D1s@D1s.
Теперь состояние и его вариации используют одни и те же ops.D2x/ops.D2s.
Алгоритм полушагов МВИ и CUF не менялись.

Старые результаты KL-МВИ, рассчитанные до v15, следует пересчитать. Малый
старый Rproj не подтверждает их сходимость после исправления. Подробности и
числа: RELEASE_NOTES_KL_MVI_D2_FIX_V15.txt и V15_VERIFICATION_RESULTS.json.
