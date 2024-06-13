# General code
### Creating DataFrames

```r
df <- data.frame(x = 1:5, y = c(6, 7, 8, 9, 10))
```
```python
df = pd.DataFrame({'x': range(1, 6), 'y': [6, 7, 8, 9, 10]})
```
### Basic Operations on DataFrames
Accessing Columns

```r
df$x
df[["x"]
```
```python
df['x']
df[['x']]
```
Subsetting Rows
```r
subset(df, x > 2)
```
```python
df[df['x'] > 2]
```
Adding a New Column
```r
df$z <- df$x + df$y
```
```python

df['z'] = df['x'] + df['y']
```
### Summary Statistics
```r
summary(df)
mean(df$x)
```
```python
df.describe()
df['x'].mean()
```
### Applying Functions
```r
apply(df, 2, sum) # Apply to columns
sapply(df, mean)  # Apply to each column
```
```python
df.apply(np.sum, axis=0)  # Apply to columns
df.apply(np.mean, axis=0)  # Apply to columns
```
### Merging DataFrames
```r
merge(df1, df2, by = "id")
```
```python
pd.merge(df1, df2, on='id')
```
### Handling Missing Data
```r
df[is.na(df)] <- 0
```
```python
df.fillna(0, inplace=True)
```
### Plotting
```r
plot(df$x, df$y)
hist(df$x)
```
```python
plt.plot(df['x'], df['y'])
plt.show()
plt.hist(df['x'])
plt.show()
```
### Statistical Tests
```r
t.test(df$x, df$y)
```
```python
stats.ttest_ind(df['x'], df['y'])
```
### Linear Regression
```r
model <- lm(y ~ x, data = df)
summary(model)
```
```python
import statsmodels.api as sm
X = df['x']
y = df['y']
X = sm.add_constant(X)  # Adds a constant term to the predicto```
```
```r
model = sm.OLS(y, X).fit()
print(model.summary())
```
### Reading/Writing Data
Reading CSV Files
```r
df <- read.csv("data.csv")
```
```python
df = pd.read_csv("data.csv")
```
Writing CSV Files
```r
write.csv(df, "output.csv")
```
```python
df.to_csv("output.csv", index=False)
```
### Reshaping Data
```r
library(reshape2)
df_melted <- melt(df, id.vars = 'id')
df_cast <- dcast(df_melted, id ~ variable)
```
```python
df_melted = pd.melt(df, id_vars=['id'])
df_cast = df_melted.pivot(index='id', columns='variable', values='value')
```
# More Plots
### Adding Legends

```r
plot(df$x, df$y, col='blue')
legend("topright", legend=c("Line 1"), col=c("blue"), lty=1)
```
```python
plt.plot(df['x'], df['y'], label='Line 1', color='blue')
plt.legend(loc='upper right')
plt.show()
```
### Adding Colors

```r
plot(df$x, df$y, col='red')
```
```python
plt.plot(df['x'], df['y'], color='red')
plt.show()
```
### Adding Horizontal and Vertical Lines
```r
abline(h=5, col='blue')  # Horizontal line at y=5
abline(v=3, col='red')   # Vertical line at x=3
```
```python
plt.plot(df['x'], df['y'])
plt.axhline(y=5, color='blue', linestyle='--')
plt.axvline(x=3, color='red', linestyle='--')
plt.show()
```
### Customizing Line Styles and Markers

```r
plot(df$x, df$y, type='o', pch=16, lty=2, col='green')
```
```python
plt.plot(df['x'], df['y'], marker='o', linestyle='--', color='green')
plt.show()
```
### Adding Titles and Labels
```r
plot(df$x, df$y, main="Plot Title", xlab="X-axis Label", ylab="Y-axis Label")
```
```python
plt.plot(df['x'], df['y'])
plt.title("Plot Title")
plt.xlabel("X-axis Label")
plt.ylabel("Y-axis Label")
plt.show()
```
### Adding Grid
```r
plot(df$x, df$y)
grid()
```
```python
plt.plot(df['x'], df['y'])
plt.grid(True)
plt.show()
```
### Subplots
```r
par(mfrow=c(1, 2))
plot(df$x, df$y)
plot(df$x, df$z)
```
```python
fig, axes = plt.subplots(1, 2)
axes[0].plot(df['x'], df['y'])
axes[1].plot(df['x'], df['z'])
plt.show()
```
### Saving Plots
```r
png("plot.png")
plot(df$x, df$y)
dev.off()
```
```python
plt.plot(df['x'], df['y'])
plt.savefig('plot.png')
plt.close()
```
### Scatter Plot with Custom Colors and Size
```r
plot(df$x, df$y, col=rainbow(nrow(df)), pch=19, cex=df$z)
```
```python
plt.scatter(df['x'], df['y'], c=df['z'], cmap='viridis', s=df['z']*10)
plt.colorbar()  # Show color scale
plt.show()
```
### Histograms with Custom Bins
```r
hist(df$x, breaks=10, col='grey', border='black')
```
```python
plt.hist(df['x'], bins=10, color='grey', edgecolor='black')
plt.show()
```
# Quantiles
### Calculating Quantiles
```r
quantile(df$x, probs = c(###25, ###5, ###75))
```
```python
quantiles = df['x'].quantile([###25, ###5, ###75])
print(quantiles)
```
### Visualizing Quantiles with Boxplot
```r
boxplot(df$x)
```
```python
plt.boxplot(df['x'])
plt.show()
```
### Adding Quantile Lines to Histograms
```r
hist(df$x)
abline(v=quantile(df$x, probs = c(###25, ###5, ###75)), col='red')
```
```python
plt.hist(df['x'], bins=10, color='grey', edgecolor='black')
# Add vertical lines at the quantiles
for quantile in [###25, ###5, ###75]:
    plt.axvline(df['x'].quantile(quantile), color='red', linestyle='--')
plt.show()
```
### Quantile-Quantile Plot (Q-Q Plot)
```r
qqnorm(df$x)
qqline(df$x, col='red')
```
```python
import scipy.stats as stats
stats.probplot(df['x'], dist="norm", plot=plt)
plt.show()
```
### Interquartile Range (IQR)
```r
IQR(df$x)
```
```python
iqr = stats.iqr(df['x'])
print(iqr)
```
### Calculating Quantiles for a Normal Distribution
```r
qnorm(c(###25, ###5, ###75), mean=mean(df$x), sd=sd(df$x))
```
```python
mean_x = df['x'].mean()
std_x = df['x'].std()
quantiles = stats.norm.ppf([###25, ###5, ###75], loc=mean_x, scale=std_x)
print(quantiles)
```
### Empirical Cumulative Distribution Function (ECDF)
```r
ecdf <- ecdf(df$x)
plot(ecdf, main="Empirical CDF")
```
```python
from statsmodels.distributions.empirical_distribution import ECDF
ecdf = ECDF(df['x'])
plt.step(ecdf.x, ecdf.y, where="post")
plt.xlabel('x')
plt.ylabel('ECDF')
plt.title('Empirical CDF')
plt.show()
```