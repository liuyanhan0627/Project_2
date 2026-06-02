answer_prompt = r"""Question: Find the domain of the expression $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.

Solution:
The expressions inside each square root must be non-negative. Therefore, $x - 2 \ge 0$, so $x \ge 2$, and $5 - x > 0$ because the denominator cannot be zero. Thus $x < 5$. The domain is $\boxed{[2,5)}$.




Question: If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det(\mathbf{A}\mathbf{B})$.

Solution:
The determinant of a product is the product of the determinants, so $\det(\mathbf{A}\mathbf{B}) = \det(\mathbf{A})\det(\mathbf{B}) = 2 \cdot 12 = \boxed{24}$.




Question: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
With two 20-pound weights lifted 12 times, Terrell lifts $2 \cdot 20 \cdot 12 = 480$ pounds. With two 15-pound weights lifted $n$ times, he lifts $2 \cdot 15 \cdot n = 30n$ pounds. Solving $30n = 480$ gives $n = 16$, so the answer is $\boxed{16}$.




Question: If the system of equations
$6x-4y=a$ and $6y-9x=b$ has a solution $(x,y)$ where $x$ and $y$ are both nonzero, find $\frac{a}{b}$, assuming $b$ is nonzero.

Solution:
Multiplying $6x-4y=a$ by $-\frac{3}{2}$ gives $6y-9x=-\frac{3}{2}a$. Since $6y-9x=b$, we have $b=-\frac{3}{2}a$, so $\frac{a}{b}=\boxed{-\frac{2}{3}}$."""

