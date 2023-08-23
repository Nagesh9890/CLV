# CLV

def calculate_clv(average_yearly_revenue, customer_lifespan, churn_rate):
    clv = (average_yearly_revenue * customer_lifespan) - (churn_rate * average_yearly_revenue * customer_lifespan)
    return clv

average_yearly_revenue = 1000
customer_lifespan = 5
churn_rate = 0.2  # (20% annual churn rate)

clv = calculate_clv(average_yearly_revenue, customer_lifespan, churn_rate)
print "The Customer Lifetime Value is: Rs %s" % clv

